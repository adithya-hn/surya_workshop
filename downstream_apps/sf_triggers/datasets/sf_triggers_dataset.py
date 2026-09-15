import re
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from astropy.io import fits

from workshop_infrastructure.datasets.helio import HelioNetCDFDataset

# SHARP bitmap filenames are the only place the HARP number and observation timestamp
# are recorded -- there is no sidecar index mapping catalog rows to mask files.
_MASK_FILENAME_RE = re.compile(
    r"hmi\.sharp_720s\.(?P<harp>\d+)\.(?P<date>\d{8})_(?P<time>\d{6})_TAI\.bitmap\.fits$"
)


def _scan_mask_dir(mask_dir: str) -> pd.DataFrame:
    """Parse HARP number + timestamp out of every SHARP bitmap filename in ``mask_dir``."""
    rows = []
    for path in sorted(Path(mask_dir).glob("*.bitmap.fits")):
        m = _MASK_FILENAME_RE.match(path.name)
        if m is None:
            continue
        ts = pd.Timestamp(
            f"{m['date'][:4]}-{m['date'][4:6]}-{m['date'][6:]}"
            f"T{m['time'][:2]}:{m['time'][2:4]}:{m['time'][4:]}"
        )
        rows.append({"harp": int(m["harp"]), "mask_ts": ts, "mask_path": str(path)})
    if not rows:
        raise ValueError(f"No SHARP bitmap files found in {mask_dir}")
    return pd.DataFrame(rows).sort_values("mask_ts")


class FlareDSDataset(HelioNetCDFDataset):
    """
    Template child class of HelioNetCDFDataset showing how to build a downstream dataset.
    Extends the base class with a flare-region spatial mask label aligned to the Surya index.

    All ``HelioNetCDFDataset`` keyword arguments (``index_path``, ``scalers``, ``channels``,
    ``s3_cache_dir``, etc.) are accepted via ``**kwargs`` and forwarded to the base class.
    ``load_forecast_frames`` defaults to ``False`` here (flare forecasting supplies its own
    labels, so future Surya frames are never fetched); pass it explicitly to override.

    Additional Args:
        return_surya_stack: If True (default), include the Surya image stack in the returned dict.
            Set to False to return only the flare mask label (useful for label inspection).
        max_number_of_samples: Cap the dataset length at this value. Useful for quick experiments.
        ds_flare_index_path: Path to the downstream flare catalog CSV index.
        ds_time_column: Column name in the flare index to use as the event timestamp.
        ds_time_tolerance: Maximum allowed time offset when matching Surya and DS indices
            (e.g., ``"15min"``). Unmatched entries are dropped.
        ds_match_direction: Merge direction passed to ``pd.merge_asof``. Use ``"forward"``
            for causal prediction (predict flares from prior solar state).
        mask_dir: Directory of SHARP bitmap FITS masks (one per flare event), named
            ``hmi.sharp_720s.<HARPNUM>.<YYYYMMDD>_<HHMMSS>_TAI.bitmap.fits``.
        mask_time_tolerance: Maximum allowed gap between a catalog row's event timestamp and
            a mask file's embedded timestamp when matching the two (e.g., ``"10min"``).

    Raises:
        ValueError: If ``ds_flare_index_path`` or ``mask_dir`` is not provided, if no overlap
            exists between the Surya and DS indices within the specified tolerance, or if any
            catalog row has no matching mask file within ``mask_time_tolerance``.
    """

    def __init__(
        self,
        # Downstream-specific parameters
        return_surya_stack: bool = True,
        max_number_of_samples: int | None = None,
        ds_flare_index_path: str | None = None,
        ds_time_column: str | None = None,
        ds_time_tolerance: str | None = None,
        ds_match_direction: Literal["forward", "backward", "nearest"] = "forward",
        mask_dir: str | None = None,
        mask_time_tolerance: str = "10min",
        # All HelioNetCDFDataset parameters (index_path, scalers, channels, s3_*, etc.)
        **kwargs,
    ):
        if ds_match_direction not in ["forward", "backward", "nearest"]:
            raise ValueError("ds_match_direction must be one of 'forward', 'backward', or 'nearest'")
        if mask_dir is None:
            raise ValueError("mask_dir must be provided for FlareDSDataset")

        # load_forecast_frames defaults to False here: flare forecasting supplies its
        # own labels, so future Surya frames never need to be fetched from disk/S3.
        kwargs.setdefault("load_forecast_frames", False)
        super().__init__(**kwargs)

        self.return_surya_stack = return_surya_stack

        # Load ds index and find intersection with Surya index
        if ds_flare_index_path is not None:
            self.ds_index = pd.read_csv(ds_flare_index_path)
        else:
            raise ValueError("ds_flare_index_path must be provided for FlareDSDataset")

        self.ds_index["ds_index"] = pd.to_datetime(
            self.ds_index[ds_time_column]
        ).values.astype("datetime64[ns]")
        self.ds_index.sort_values("ds_index", inplace=True)

        # Create Surya valid indices and find closest match to DS index
        self.df_valid_indices = pd.DataFrame(
            {"valid_indices": self.valid_indices}
        ).sort_values("valid_indices")
        self.df_valid_indices = pd.merge_asof(
            self.df_valid_indices,
            self.ds_index,
            right_on="ds_index",
            left_on="valid_indices",
            direction=ds_match_direction,
        )
        # Remove duplicates keeping closest match
        self.df_valid_indices["index_delta"] = np.abs(
            self.df_valid_indices["valid_indices"] - self.df_valid_indices["ds_index"]
        )
        self.df_valid_indices = self.df_valid_indices.sort_values(
            ["ds_index", "index_delta"]
        )
        self.df_valid_indices.drop_duplicates(
            subset="ds_index", keep="first", inplace=True
        )
        # Enforce a maximum time tolerance for matches
        if ds_time_tolerance is not None:
            self.df_valid_indices = self.df_valid_indices.loc[
                self.df_valid_indices["index_delta"] <= pd.Timedelta(ds_time_tolerance),
                :,
            ]
            if len(self.df_valid_indices) == 0:
                raise ValueError("No intersection between Surya and DS indices")

        # Match each catalog row to its flare-region mask file by nearest timestamp.
        # self.df_valid_indices is sorted ascending by "ds_index" at this point, which
        # merge_asof requires.
        mask_index = _scan_mask_dir(mask_dir)
        self.df_valid_indices = pd.merge_asof(
            self.df_valid_indices,
            mask_index,
            left_on="ds_index",
            right_on="mask_ts",
            direction="nearest",
            tolerance=pd.Timedelta(mask_time_tolerance),
        )
        unmatched = self.df_valid_indices["mask_path"].isna()
        if unmatched.any():
            bad = self.df_valid_indices.loc[unmatched, "ds_index"].tolist()
            raise ValueError(
                f"No mask file within {mask_time_tolerance} for ds_index timestamps: {bad}"
            )

        # Override valid indices variables to reflect matches between Surya and DS
        self.valid_indices = [
            pd.Timestamp(date) for date in self.df_valid_indices["valid_indices"]
        ]
        self.adjusted_length = len(self.valid_indices)
        self.df_valid_indices.set_index("valid_indices", inplace=True)

        if max_number_of_samples is not None and max_number_of_samples < self.adjusted_length:
            self.valid_indices = self.valid_indices[:max_number_of_samples]
            self.df_valid_indices = self.df_valid_indices.iloc[:max_number_of_samples]
            self.adjusted_length = max_number_of_samples

    def __len__(self):
        return self.adjusted_length

    def _load_mask(self, path: str) -> np.ndarray:
        """Load a SHARP bitmap FITS file as a binary (1, H, W) float32 mask."""
        with fits.open(path) as hdul:
            data = np.asarray(hdul[0].data)
        mask = np.nan_to_num(data, nan=0.0)
        mask = (mask > 0).astype(np.float32)  # binarize defensively
        return mask[None, :, :]  # (1, H, W) -- channel-first, matches ts/forecast convention

    def __getitem__(self, idx: int) -> dict:
        """
        Args:
            idx: Dataset index.

        Returns:
            Dictionary containing:
                forecast (np.float32): Binary flare-region mask, shape (1, H, W).
                ds_index (str): ISO-format timestamp from the flare index.
            When ``return_surya_stack=True``, also includes all keys from
            ``HelioNetCDFDataset.__getitem__`` (ts, time_delta_input, lead_time_delta, etc.).
        """
        sample = super().__getitem__(idx=idx) if self.return_surya_stack else {}
        mask_path = self.df_valid_indices.iloc[idx]["mask_path"]
        sample["forecast"] = self._load_mask(mask_path)
        sample["ds_index"] = self.df_valid_indices["ds_index"].iloc[idx].isoformat()
        return sample

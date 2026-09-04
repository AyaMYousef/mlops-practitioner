from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration loaded from environment variables."""

    project_root: Path = Path(__file__).resolve().parents[2]

    data_filename: str = "green_tripdata_2024-01.parquet"
    model_filename: str = "baseline.pkl"

    random_state: int = 42
    n_estimators: int = 100
    max_depth: int | None = None

    test_size: float = 0.2

    host: str = "0.0.0.0"
    port: int = 8000

    model_config = SettingsConfigDict(
        env_prefix="PRODML_",
        env_file=".env",
        extra="ignore",
    )

    @property
    def data_path(self) -> Path:
        """Return the path to the training data."""
        return self.project_root / "data" / "raw" / self.data_filename

    @property
    def model_path(self) -> Path:
        """Return the path to the persisted model."""
        return self.project_root / "models" / self.model_filename


settings = Settings()

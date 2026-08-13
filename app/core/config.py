from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    groq_key: SecretStr
    model: str
    database_url: str = "postgresql://postgres:postgres@localhost:5432/stateflow"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()  # type: ignore

import json
import pandas as pd


class LoadAndGet:

    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None

    def load_file(self):
        self.df = pd.read_csv(self.file_path)
        return self.df

    def get_profile(self):

        if self.df is None:
            self.load_file()

        df = self.df

        profile = {
            "dataset": {
                "rows": df.shape[0],
                "columns": df.shape[1]
            },

            "columns": df.columns.tolist(),

            "dtypes": df.dtypes.astype(str).to_dict(),

            "missing_values": df.isnull().sum().to_dict(),

            "unique_values": df.nunique().to_dict(),

            "head": df.head().to_dict(orient="records"),

            "describe": (
                df.describe(include="all")
                .fillna("")
                .to_dict()
            )
        }

        return profile, df

    def get_profile_json(self):

        profile, df = self.get_profile()

        profile_json = json.dumps(
            profile,
            indent=4,
            default=str
        )

        return profile_json, df
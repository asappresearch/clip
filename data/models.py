import datetime
import os
import pickle
from dataclasses import dataclass
import typing as t

import boto3
import pandas as pd

DEBUG = {"True": True, "False": False, None: False}.get(os.environ.get("DEBUG"), False)
print("DEBUG: ", DEBUG, type(DEBUG))


@dataclass
class Span:
    id: str
    document_id: str
    type: str
    start: int
    end: int


@dataclass
class UserSpans:
    user_id: str
    spans: t.List[Span]
    notes: str


@dataclass
class Document:
    filename: str
    id: str
    text: str
    users: t.List[UserSpans]
    is_finalized: bool

    @staticmethod
    def get_path(filename: str, id: int):
        return f"revised_training_data/{id}.pkl"

    def save(self, s3_prefix: t.Optional["str"] = "gold_standard"):
        path = self.get_path(self.filename, self.id)
        os.makedirs(os.path.split(path)[0], exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def from_json(cls, json_: t.Dict):
        users = [
            UserSpans(
                user_id=user["user_id"],
                notes=user["notes"],
                spans=[Span(**span) for span in user["spans"]],
            )
            for user in json_["users"]
        ]
        return cls(
            filename=json_["filename"],
            id=json_["id"],
            text=json_["text"],
            users=users,
            is_finalized=json_["is_finalized"],
        )

    @classmethod
    def load(cls, filename: str, id_: int):
        path = cls.get_path(filename, id_)
        os.makedirs(os.path.split(path)[0], exist_ok=True)

        if os.path.exists(path):
            with open(path, "rb") as f:
                return pickle.load(f)

        df = pd.read_csv(f'{filename}/{id_}.pkl')
        documents = df[df.document_id == id_]
        user_docs = []
        for _, doc in documents.iterrows():
            spans = []
            for span in eval(doc.annotations):
                spans.append(
                    Span(
                        id=span[0],
                        document_id=doc.document_id,
                        type=span[1],
                        start=span[2],
                        end=span[2] + span[3],
                    )
                )

            user_docs.append(
                UserSpans(
                    user_id=-1,
                    spans=spans,
                    notes=doc.user_notes if not pd.isnull(doc.user_notes) else None,
                )
            )
        tokens = eval(documents.iloc[0].document)["tokens"]
        text = "".join([x[0] for x in tokens])
        doc = cls(
            filename=filename, id=id_, text=text, users=user_docs, is_finalized=False
        )
        doc.save()
        return doc

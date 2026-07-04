# -*- coding: utf-8 -*-

"""
AnnoMI Annotated Transcript Converter

Merges the AnnoMI full CSV with a per-transcript annotated CSV (containing Co_cat5 labels),
groups consecutive utterances by the same speaker, and outputs a combined JSON.

Output format:
  [ [ {uttr_turn_1_uttr_1}, ... ], [ {uttr_turn_2_uttr_1}, ... ], ... ]
  Each inner list is one speaker-turn (one or more consecutive utterances by the same speaker).

Utterance dict fields:
  - speaker  : "Counselor" | "Client"
  - category : Co_cat5 index string for Counselor; "-1" for Client
  - utterance: cleaned utterance text (bracket annotations removed)
  - csv_idx  : utterance_id in the original CSV
  - AnnoMI_transcriptId_utteranceId: "<transcript_id>_<utterance_id>"

Command: uv run python -m src.make_dataset.convert_AnnoMI --raw_full_csv /app/dataset/raw/AnnoMI_Co5category_annotated/AnnoMI-full.csv --target_id 36 --annotated_csv /app/dataset/raw/AnnoMI_Co5category_annotated/AnnoMI_annotated_36.csv --out_json /app/dataset/json/AnnoMI/36.json
"""

import re
import copy
import json
import logging

import pandas as pd

CO_CATEGORY2IDX: dict[str, str] = {
    "q": "0",
    "af": "1",
    "ref": "2",
    "sum": "3",
    "other": "4",
}


class AnnotatedTranscriptConverter:
    def __init__(self) -> None:
        self.log = logging.getLogger(__name__)

    def save_json(self, out_json_path: str, obj) -> None:
        with open(out_json_path, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=4, ensure_ascii=False)

    def merge_raw_and_annotated(
        self,
        target_id: int,
        raw_full_csv_path: str,
        annotated_csv_path: str,
    ) -> pd.DataFrame:
        """Filter the full CSV by target_id, then left-join the annotated CSV on utterance_id.

        The annotated CSV must contain at least: utterance_id, Co_cat5.
        Speakers are not merged at this stage.
        """
        raw_df = pd.read_csv(raw_full_csv_path, dtype={"utterance_text": str})
        tgt_df = raw_df[raw_df["transcript_id"] == target_id].copy()

        annotated_df = pd.read_csv(annotated_csv_path, encoding="utf-8-sig")
        merge_df = tgt_df.merge(
            annotated_df[["utterance_id", "Co_cat5"]],
            on="utterance_id",
            how="left",
        )
        return merge_df

    def _make_uttr_dict(
        self, df_row, index: int, co_category2idx: dict
    ) -> dict:
        interlocutor = df_row["interlocutor"]
        speaker = "Counselor" if interlocutor == "therapist" else "Client"

        if speaker == "Counselor":
            co_cat = df_row["Co_cat5"]
            if co_cat is None or (isinstance(co_cat, float) and pd.isna(co_cat)):
                self.log.error(f"Co_cat5 is empty at index {index}, utterance_id={df_row['utterance_id']}")
                raise ValueError(f"Co_cat5 missing for Counselor at utterance_id={df_row['utterance_id']}")
            category = co_category2idx.get(str(co_cat).strip(), "-2")
        else:
            category = "-1"

        utterance = re.sub(r"\[.*?\]", "", str(df_row["utterance_text"]))

        return {
            "speaker": speaker,
            "category": category,
            "utterance": utterance,
            "csv_idx": df_row["utterance_id"],
            "AnnoMI_transcriptId_utteranceId": f"{df_row['transcript_id']}_{df_row['utterance_id']}",
        }

    def merge_same_speaker(
        self, merge_df: pd.DataFrame, co_category2idx: dict
    ) -> list:
        """Group consecutive same-speaker utterances into turns.

        Returns:
            list of turns, each turn is a list of utterance dicts.
        """
        combined_turns = []
        current_speaker = None
        current_turn = []

        for index, row in merge_df.iterrows():
            d = self._make_uttr_dict(row, index, co_category2idx)

            if current_speaker is None:
                current_speaker = d["speaker"]
                current_turn.append(d)
                continue

            if d["speaker"] == current_speaker:
                current_turn.append(d)
            else:
                combined_turns.append(current_turn)
                current_turn = [d]
                current_speaker = d["speaker"]

        if current_turn:
            combined_turns.append(current_turn)

        return copy.deepcopy(combined_turns)


def convert_annotated_transcript(
    raw_full_csv: str,
    target_id: int,
    annotated_csv: str,
    out_json_path: str,
    co_category2idx: dict = CO_CATEGORY2IDX,
) -> list:
    """Convert one annotated AnnoMI transcript to a speaker-turn JSON file.

    Args:
        raw_full_csv   : Path to AnnoMI-full.csv.
        target_id      : transcript_id to extract.
        annotated_csv  : Path to the per-transcript annotated CSV (must have Co_cat5 column).
        out_json_path  : Destination path for the output JSON.
        co_category2idx: Mapping from Co_cat5 string label to index string.

    Returns:
        combined_turns (list): the same data written to out_json_path.
    """
    converter = AnnotatedTranscriptConverter()
    merge_df = converter.merge_raw_and_annotated(target_id, raw_full_csv, annotated_csv)
    combined_turns = converter.merge_same_speaker(merge_df, co_category2idx)
    converter.save_json(out_json_path, combined_turns)
    print(f"Saved: {out_json_path}  ({len(combined_turns)} turns)")
    return combined_turns


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert an annotated AnnoMI transcript to JSON.")
    parser.add_argument("--raw_full_csv", required=True, help="Path to AnnoMI-full.csv")
    parser.add_argument("--target_id", type=int, required=True, help="transcript_id to process")
    parser.add_argument("--annotated_csv", required=True, help="Path to the annotated CSV for the target transcript")
    parser.add_argument("--out_json", required=True, help="Output JSON path")
    args = parser.parse_args()

    convert_annotated_transcript(
        raw_full_csv=args.raw_full_csv,
        target_id=args.target_id,
        annotated_csv=args.annotated_csv,
        out_json_path=args.out_json,
    )

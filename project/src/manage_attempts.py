"""Review recognition attempts without conflating target and ground truth."""
from __future__ import annotations

import argparse
from pathlib import Path

import config as C
from attempt_store import RecognitionAttemptStore


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=str, default=str(C.RECOGNITION_ATTEMPTS_DB))
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("pending", help="list attempts awaiting ground-truth review")
    verify = subparsers.add_parser("verify", help="record independently verified ground truth")
    verify.add_argument("--attempt-id", type=int, required=True)
    verify.add_argument("--ground-truth", required=True)
    verify.add_argument("--verified-by", required=True)
    correct = subparsers.add_parser("correct", help="correct verified ground truth with audit")
    correct.add_argument("--attempt-id", type=int, required=True)
    correct.add_argument("--ground-truth", required=True)
    correct.add_argument("--verified-by", required=True)
    correct.add_argument("--reason", required=True)
    history = subparsers.add_parser("history", help="show ground-truth review history")
    history.add_argument("--attempt-id", type=int, required=True)
    args = parser.parse_args()

    db_path = Path(args.db)
    if not db_path.is_absolute():
        db_path = C.ROOT / db_path
    store = RecognitionAttemptStore(db_path)
    if args.command == "pending":
        rows = store.pending_attempts()
        if not rows:
            print("No attempts are awaiting review.")
            return
        for row in rows:
            print(
                f"#{row['attempt_id']} target={row['target_label']} "
                f"prediction={row['predicted_label']} confidence={row['confidence']:.3f}"
            )
        return

    if args.command == "history":
        rows = store.review_history(args.attempt_id)
        if not rows:
            print(f"No review history for attempt #{args.attempt_id}.")
            return
        for row in rows:
            print(
                f"{row['reviewed_at']} {row['action']}: "
                f"{row['previous_ground_truth_label']} -> {row['new_ground_truth_label']} "
                f"({row['previous_correctness_status']} -> "
                f"{row['new_correctness_status']}) by {row['reviewer']}; "
                f"reason={row['reason'] or '-'}"
            )
        return

    if args.command == "correct":
        status = store.correct_ground_truth(
            args.attempt_id, args.ground_truth, args.verified_by, args.reason
        )
    else:
        status = store.verify_ground_truth(
            args.attempt_id, args.ground_truth, args.verified_by
        )
    print(f"Attempt #{args.attempt_id} reviewed: {status}")


if __name__ == "__main__":
    main()

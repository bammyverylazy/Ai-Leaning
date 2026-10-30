import threading
import time
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from speech import SpeechService, spoken_text


class SpeechServiceTests(unittest.TestCase):
    def test_canonical_labels_are_naturalized_for_speech(self):
        self.assertEqual(spoken_text("CHRISTMAS (variant 1)"), "Christmas")
        self.assertEqual(spoken_text("WHAT FOR"), "What For")
        self.assertEqual(spoken_text("HURDLE/TRIP"), "Hurdle or Trip")

    def test_speaks_only_enqueued_committed_labels_and_respects_mute(self):
        spoken = []
        spoken_event = threading.Event()
        started = []

        def fake_speaker(text, cancelled):
            if not cancelled.is_set():
                spoken.append(text)
                spoken_event.set()

        service = SpeechService(
            lambda _error: None,
            speaker=fake_speaker,
            on_started=lambda text, timestamp: started.append((text, timestamp)),
        )
        try:
            service.set_enabled(False)
            self.assertFalse(service.enqueue("HELLO"))
            service.set_enabled(True)
            self.assertTrue(service.enqueue("THANK YOU"))
            self.assertTrue(spoken_event.wait(1))
            service.set_enabled(False)
            time.sleep(0.05)

            self.assertEqual(spoken, ["Thank You"])
            self.assertEqual([text for text, _ in started], ["Thank You"])
            self.assertGreater(started[0][1], 0)
        finally:
            service.close()

    def test_muting_cancels_active_and_queued_speech(self):
        started = threading.Event()
        cancelled_during_speech = threading.Event()
        spoken = []

        def fake_speaker(text, cancelled):
            spoken.append(text)
            started.set()
            if cancelled.wait(1):
                cancelled_during_speech.set()

        service = SpeechService(lambda _error: None, speaker=fake_speaker)
        try:
            service.enqueue("HELLO")
            self.assertTrue(started.wait(1))
            service.enqueue("THANK YOU")
            service.set_enabled(False)
            self.assertTrue(cancelled_during_speech.wait(1))
            time.sleep(0.05)

            self.assertEqual(spoken, ["Hello"])
        finally:
            service.close()


if __name__ == "__main__":
    unittest.main()

import hashlib
import tempfile
import unittest
from pathlib import Path

from scripts.prefetch_weights import transfer


class TransferTests(unittest.TestCase):
    def test_verified_atomic_promotion(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            data = b"offline fixture, not model weights"
            source = root / "source"
            source.write_bytes(data)
            target, temporary = root / "model/weights", root / "tmp/partial"
            result = transfer(source.as_uri(), target, temporary, len(data), hashlib.sha256(data).hexdigest())
            self.assertEqual(target.read_bytes(), data)
            self.assertFalse(temporary.exists())
            self.assertEqual(result["bytes"], len(data))

    def test_bad_hash_cannot_replace_existing_valid_file(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source, target = root / "source", root / "target"
            source.write_bytes(b"bad")
            target.write_bytes(b"existing")
            with self.assertRaises(ValueError):
                transfer(source.as_uri(), target, root / "partial", 3, "0" * 64)
            self.assertEqual(target.read_bytes(), b"existing")


if __name__ == "__main__":
    unittest.main()

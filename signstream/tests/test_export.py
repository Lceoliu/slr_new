from signstream.inference.rle import run_length_encode


def test_rle():
    seq = [1, 1, 1, 2, 2, 3]
    assert run_length_encode(seq) == [(1, 3), (2, 2), (3, 1)]

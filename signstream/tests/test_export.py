from signstream.inference.rle import run_length_encode


def test_rle_encoding():
    seq = [[1], [1], [2], [2], [2], [3]]
    encoded = run_length_encode(seq)
    assert encoded == [[1], ["NC", 1], [2], ["NC", 2], [3]]

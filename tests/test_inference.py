import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1] / 'src'))
from inference import predict_ticket

def test_prediction_shape():
    result = predict_ticket('I was charged twice and need a refund.')
    assert result['category'] in {'billing','technical','account','product'}
    assert result['urgency'] in {'low','medium','high'}
    assert 'queue' in result

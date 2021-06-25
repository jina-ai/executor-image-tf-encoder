__copyright__ = "Copyright (c) 2020-2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

from jina import Flow, Document
from jinahub.encoder.image_tf_encoder import ImageTFEncoder


def test_exec():
    pass

    '''
    
    f = Flow().add(uses=ImageTFEncoder)
    with f:
        resp = f.post(on='/test', inputs=Document(), return_results=True)
        assert resp is not None
    '''

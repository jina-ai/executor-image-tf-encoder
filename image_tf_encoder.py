__copyright__ = "Copyright (c) 2020-2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

from typing import Union, Iterable, List, Any, Optional, Dict

import numpy as np
from jina import DocumentArray, Executor, requests
from jina.logging.logger import JinaLogger


def _batch_generator(data: List[Any], batch_size: Optional[int]):
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]


class ImageTFEncoder(Executor):
    """
    :class:`ImageKerasEncoder` encodes ``Document`` content from a ndarray,
    potentially B x (Channel x Height x Width) into a ndarray of `B x D`.

    Where `B` is the batch size and `D` is the Dimension.

    The :class:`ImageKerasEncoder` wraps the models from
    `tensorflow.keras.applications`. <https://keras.io/applications/>`_.

    :param model_name: the name of the model. Supported models include
        ``DenseNet121``, ``DenseNet169``, ``DenseNet201``,
        ``InceptionResNetV2``, ``InceptionV3``, ``MobileNet``,
        ``MobileNetV2``, ``NASNetLarge``, ``NASNetMobile``,
        ``ResNet101``, ``ResNet152``, ``ResNet50``, ``ResNet101V2``,
        ``ResNet152V2``, ``ResNet50V2``, ``VGG16``, ``VGG19``,
        ``Xception``,
    :param img_shape: The shape of the image to be encoded.
    :param pool_strategy: the pooling strategy. Options are:
        - `None`: Means that the output of the model will be the 4D tensor
            output of the last convolutional block.
        - `avg`: ;eans that global average pooling will be applied to the
            output of the last convolutional block, and thus the output of
            the model will be a 2D tensor.
        - `max`: Means that global max pooling will be applied.
    :param default_batch_size: size of each batch
    :param default_traversal_paths: traversal path of the Documents, (e.g. 'r', 'c')
    :param on_gpu: set to True if using GPU
    :param args: additional positional arguments.
    :param kwargs: additional positional arguments.
    """

    def __init__(self,
                 model_name: str = 'MobileNetV2',
                 img_shape: int = 336,
                 pool_strategy: str = 'max',
                 default_batch_size: int = 32,
                 default_traversal_paths: List[str] = ['r'],
                 on_gpu: bool = True,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.model_name = model_name
        self.pool_strategy = pool_strategy
        self.img_shape = img_shape
        self.default_batch_size = default_batch_size
        self.default_traversal_paths = default_traversal_paths
        self.on_gpu = on_gpu
        self.logger = JinaLogger(self.__class__.__name__)

        import tensorflow as tf
        cpus = tf.config.experimental.list_physical_devices(device_type='CPU')
        gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
        if self.on_gpu and len(gpus) > 0:
            cpus.append(gpus[0])
        if self.on_gpu and len(gpus) == 0:
            self.logger.warning('You tried to use a GPU but no GPU was found on'
                                ' your system. Defaulting to CPU!')
        tf.config.experimental.set_visible_devices(devices=cpus)

        model = getattr(tf.keras.applications, self.model_name)(
            input_shape=(self.img_shape, self.img_shape, 3),
            include_top=False,
            pooling=self.pool_strategy,
            weights='imagenet')
        model.trainable = False
        self.model = model

    @requests
    def encode(self, docs: DocumentArray, parameters: Dict, **kwargs):
        """
        Encode document content into a ndarray of `B x D`. `
        B` is the batch size and `D` is the Dimension.

        :param content: Image to be encoded, expected a `np.ndarray` of
            BatchSize x (Channel x Height x Width).
        :param args: additional positional arguments.
        :param kwargs: additional positional arguments.
        :return: Encoded result as a `BatchSize x D` numpy ``ndarray``,
            `D` is the output dimension
        """
        if docs:
            document_batches_generator = self._get_input_data(docs, parameters)
            self._create_embeddings(document_batches_generator)

    def _create_embeddings(self, document_batches_generator: Iterable):
        for document_batch in document_batches_generator:
            blob_batch = np.stack([d.blob for d in document_batch])
            embedding_batch = self.model(blob_batch)
            for document, embedding in zip(document_batch, embedding_batch):
                document.embedding = np.array(embedding)

    def _get_input_data(self, docs: DocumentArray, parameters: dict):
        traversal_paths = parameters.get('traversal_paths', self.default_traversal_paths)
        batch_size = parameters.get('batch_size', self.default_batch_size)

        # traverse thought all documents which have to be processed
        flat_docs = docs.traverse_flat(traversal_paths)

        # filter out documents without images
        filtered_docs = [doc for doc in flat_docs if doc.blob is not None]

        return _batch_generator(filtered_docs, batch_size)

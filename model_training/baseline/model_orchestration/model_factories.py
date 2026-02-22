from classes import (
    ModelFactory
)
from ..naive import NaiveImpl
from ..mean import MeanImpl

class NaiveFactory(ModelFactory):
    def create(self) -> NaiveImpl:
        return NaiveImpl()

class MeanFactory(ModelFactory):
    def create(self) -> MeanImpl:
        return MeanImpl()
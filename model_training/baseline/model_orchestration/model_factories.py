from classes import (
    ModelFactory, Organizer, SeasonalityStrategy
)
from ..naive import NaiveImpl
from ..mean import MeanImpl

class NaiveFactory(ModelFactory):
    def create(self, organizer: Organizer, strategy: SeasonalityStrategy) -> NaiveImpl:
        return NaiveImpl(organizer, strategy)

class MeanFactory(ModelFactory):
    def create(self, organizer: Organizer, strategy: SeasonalityStrategy) -> MeanImpl:
        return MeanImpl(organizer, strategy)
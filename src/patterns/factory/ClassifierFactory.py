from patterns.strategy.AdaBoostStrategy import AdaBoostStrategy
from patterns.strategy.VotingStrategy import VotingStrategy
from patterns.strategy.SgdStrategy import SgdStrategy
from patterns.strategy.Hist_gbStrategy import Hist_gbStrategy
from patterns.strategy.RandomTreesStrategy import RandomTreesStrategy

#Returns the startegy the user selected in main
class ClassifierFactory:
    def get_strategy(self, choice):
        """Return the appropriate strategy based on user input."""
        if choice == "1":
            return AdaBoostStrategy()
        elif choice == "2":
            return VotingStrategy()
        elif choice == "3":
            return SgdStrategy()
        elif choice == "4":
            return Hist_gbStrategy()
        elif choice == "5":
            return RandomTreesStrategy()
        else:
            raise ValueError("Invalid choice. Pick 1, 2, 3, 4, or 5.")

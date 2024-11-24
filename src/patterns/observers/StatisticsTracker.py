from collections import defaultdict
from patterns.observers.observer import Observer, ClassificationEvent

class StatisticsTracker(Observer):
    def __init__(self):
        self.class_counts = defaultdict(int)

    def update(self, event: ClassificationEvent):
        self.class_counts[event.predicted_class] += 1

    def display_stats(self):
        print("Classification Counts:")
        for cls, count in self.class_counts.items():
            print(f"{cls}: {count}")

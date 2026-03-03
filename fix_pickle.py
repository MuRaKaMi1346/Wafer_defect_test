import pickle
import pandas as pd

class RenameUnpickler(pickle._Unpickler):
    def find_class(self, module, name):

        if module.startswith("pandas.indexes"):
            module = module.replace("pandas.indexes", "pandas.core.indexes")

        if module.startswith("pandas.tslib"):
            module = module.replace("pandas.tslib", "pandas._libs.tslibs")

        return super().find_class(module, name)

def load_old_pickle(file):
    return RenameUnpickler(file, encoding="latin1").load()

print("Loading old dataset ...")
with open("wafer_test_project\LSWMD.pkl", "rb") as f:
    df = load_old_pickle(f)

print("Saving fixed dataset ...")
df.to_pickle("LSWMD_fixed.pkl")

print("DONE ✔")
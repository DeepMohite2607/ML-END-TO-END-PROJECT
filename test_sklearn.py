print("Testing sklearn import...")
try:
    import sklearn
    print(f"sklearn version: {sklearn.__version__}")
    print("sklearn imported successfully!")
    
    from sklearn.model_selection import train_test_split
    print("train_test_split imported successfully!")
    
    print("\nAll imports successful!")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

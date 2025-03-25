def test_imports():
    # Test core dependencies
    from autogluon.tabular import TabularPredictor
    import box
    import numpy
    import pandas
    import sklearn
    import torch
    
    # Test project-specific imports
    from src.conf.conf import get_config
    from src.conf.environment import activate_env
    from src.models import autogluon as project_autogluon
    
    print("All core dependencies imported successfully")
    
    # Test project configuration
    cfg = get_config()
    print("Project configuration loaded successfully")
    
    # Test environment activation
    activate_env()
    print("Environment activated successfully")
    
    print("All requirements satisfied")

if __name__ == "__main__":
    test_imports()
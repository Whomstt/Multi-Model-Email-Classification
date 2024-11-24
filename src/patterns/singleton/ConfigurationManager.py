class ConfigurationManager:
    _instance = None

    
    _config_value = None

    #Allows only one instance
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConfigurationManager, cls).__new__(cls)
            cls._instance._config_value = "configuration"
        return cls._instance

    
    def get_config_value(self):
        return self._config_value

    
    def set_config_value(self, value):
        self._config_value = value

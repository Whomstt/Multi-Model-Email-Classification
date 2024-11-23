class ConfigurationManager:
    # 1. Private static instance variable
    _instance = None

    # 2. Shared configuration data (example properties)
    _config_value = None

    # 3. Private constructor to prevent instantiation from outside
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConfigurationManager, cls).__new__(cls)
            # Initialize the configuration (example)
            cls._instance._config_value = "Some configuration data"
        return cls._instance

    # 4. Method to get the configuration value
    def get_config_value(self):
        return self._config_value

    # 5. Method to set the configuration value
    def set_config_value(self, value):
        self._config_value = value

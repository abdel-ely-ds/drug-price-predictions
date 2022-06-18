class NoModeSpecified(Exception):
    """Exception raised for errors in no flag was specified
    Attributes:
        message -- explanation of the error
    """

    def __init__(self, message="please specify one of the flags train or predict"):
        self.message = message
        super().__init__(self.message)


class MultipleModes(Exception):
    """Exception raised for errors in multiple flags were specified
    Attributes:
        message -- explanation of the error
    """

    def __init__(
        self,
        message="please choose only one of flags train or predict not both at the same time",
    ):
        self.message = message
        super().__init__(self.message)

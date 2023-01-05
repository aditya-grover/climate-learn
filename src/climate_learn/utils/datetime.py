Year = int
"""A type definition for representing years."""


class Days:
    """A data object that represents a number of days.

    :param value: A number of days.
    :type value: int|float
    """

    def __init__(self, value):
        """Constructor method"""
        self.value = value

    def days(self):
        """Getter method.

        :return: The number of days represented by this object.
        :rtype: int|float
        """
        return self.value

    def hours(self):
        """Getter method.

        :return: The number of hours represented by this object.
        :rtype: int
        """
        return int(self.value * 24)


class Hours:
    """A data object that represents a number of hours.

    :param value: A number of hours.
    :type value: int|float
    """

    def __init__(self, value):
        """Constructor method"""
        self.value = value

    def days(self):
        """Getter method.

        :return: The number of days represented by this object.
        :rtype: int
        """
        return self.value // 24

    def hours(self):
        """Getter method.

        :return: The number of hours represented by this object.
        :rtype: int|float
        """
        return self.value

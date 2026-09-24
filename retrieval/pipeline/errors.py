class StageError(Exception):
    """A pipeline stage failed. Carries a machine-readable stage/code so the
    caller can return a clean result instead of letting the LLM answer
    without evidence."""

    def __init__(self, stage, code, message):
        super().__init__(message)
        self.stage = stage
        self.code = code
        self.message = message

    def to_dict(self):
        return {"stage": self.stage, "code": self.code, "message": self.message}


def error_result(stage, code, message):
    return {"success": False, "error": {"stage": stage, "code": code, "message": message}}

"""Handle user-specified and default parameters."""


# -----------------------------------------------------------------------------
# Python script <==> dictionaries conversion
# -----------------------------------------------------------------------------
def python_to_params(script_contents):
    params = {}
    exec script_contents in {}, params
    return params
    
def params_to_python(params):
    return "\n".join(["{0:s} = {1:s}".format(key, str(val))
        for key, val in sorted(params.iteritems())])
    

# -----------------------------------------------------------------------------
# Parameter functions
# -----------------------------------------------------------------------------
    
def get_params(params_user):
    pass


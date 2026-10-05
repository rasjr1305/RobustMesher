def parallel_print(string: str, comm=None):
    """Print a string once from appropriate rank.

    Prints the string only once: from rank 0 if no ensemble_comm, or from
    ensemble rank 0 and comm rank 0 if ensemble_comm is present.

    Parameters
    ----------
    string : str
        The string to print.
    comm : Firedrake.ensemble_communicator, optional
        A Firedrake ensemble communicator or standard MPI communicator.
    """
    if comm is None:
        print(string, flush=True)
    else:
        if getattr(comm, "ensemble_comm", None) is not None:
            if comm.ensemble_comm.rank == 0 and comm.comm.rank == 0:
                print(string, flush=True)
        elif getattr(comm, "rank", None) is not None:
            if comm.rank == 0:
                print(string, flush=True)



def run_in_one_core(func):
    """Decorate to execute function only on rank 0.

    Ensures the decorated function runs only on the root process (rank 0)
    of the communicator. Other processes skip execution. Useful for I/O
    operations and serial tasks in parallel environments.

    Parameters
    ----------
    func : callable
        Function to decorate. The first argument of func must be an object
        with a `comm` attribute containing an MPI communicator.

    Returns
    -------
    callable
        Wrapped function that executes only on rank 0.

    Notes
    -----
    The function checks for two types of communicators:
    - Ensemble communicator: Runs only if both `ensemble_comm.rank` == 0
      and comm.rank == 0.
    - Regular communicator: Runs only if comm.rank == 0.
    - If comm is None, the function runs normally without restrictions.

    The function does not broadcast results to other processes.

    See Also
    --------
    run_in_one_core_and_broadcast : Similar decorator that also broadcasts results.

    Examples
    --------
    >>> @run_in_one_core
    ... def save_file(obj, filename):
    ...     # Only rank 0 writes the file
    ...     with open(filename, 'w') as f:
    ...         f.write(str(obj.data))
    """

    def wrapper(*args, **kwargs):
        comm = args[0].comm
        if comm is None:
            return func(*args, **kwargs)
        else:
            if getattr(comm, "ensemble_comm", None) is not None:
                if comm.ensemble_comm.rank == 0 and comm.comm.rank == 0:
                    return func(*args, **kwargs)
            elif getattr(comm, "rank", None) is not None:
                if comm.rank == 0:
                    return func(*args, **kwargs)

    return wrapper

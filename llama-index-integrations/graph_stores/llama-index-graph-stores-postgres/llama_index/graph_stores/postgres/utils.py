from sqlalchemy.orm import Session
from sqlalchemy import Engine, exc, sql


def check_db_availability(engine: Engine, check_vector: bool = False) -> None:
    try:
        with engine.connect() as conn:
            if check_vector:
                conn.execute(sql.text("""SELECT '[1]'::vector;"""))
            else:
                conn.execute(sql.text("""SELECT 1;"""))
from sqlalchemy.orm import Session
from sqlalchemy import Engine, exc, sql

# Import more specific exception types
from sqlalchemy.exc import OperationalError, ProgrammingError

def check_db_availability(engine: Engine, check_vector: bool = False) -> None:
    try:
        with engine.connect() as conn:
            if check_vector:
                conn.execute(sql.text("""SELECT '[1]'::vector;"""))
            else:
                conn.execute(sql.text("""SELECT 1;"""))
    except OperationalError as e:
        if hasattr(e, 'orig') and hasattr(e.orig, 'args') and len(e.orig.args) > 0:
            db_error_code = e.orig.args[0]
            if db_error_code == 28000:  # Authentication error
                raise ValueError(
                    "Could not connect to the PostgreSQL server. "
                    "Please check if the connection string is correct."
                ) from e
            else:
                raise ValueError(
                    "An error occurred while checking the database availability."
                ) from e
    except ProgrammingError as e:
        if "vector" in str(e):
            raise ValueError(
                "Please confirm if your PostgreSQL supports vector search. "
                "You can check this by running the query `SELECT '[1]'::vector` in PostgreSQL."
            ) from e
        else:
            raise ValueError(
                "An error occurred while checking the database availability."
            ) from e
        if hasattr(e, 'orig') and hasattr(e.orig, 'pgcode'):
            if e.orig.pgcode == '42883':  # undefined_function
                raise ValueError(
                    "Please confirm if your PostgreSQL supports vector search. "
                    "You can check this by running the query `SELECT '[1]'::vector` in PostgreSQL."
                ) from e
            elif e.orig.pgcode == '28P01':  # invalid_password
                raise ValueError(
                    "Could not connect to the PostgreSQL server. "
                    "Please check if the connection string is correct."
                ) from e
            else:
                raise ValueError(
                    f"An error occurred while checking the database availability: {e.orig.pgcode}"
                ) from e
        else:
            raise ValueError(
                "An error occurred while checking the database availability."
            ) from e


def get_or_create(session: Session, model, **kwargs):
    instance = session.query(model).filter_by(**kwargs).first()
    if instance:
        return instance, False
    else:
        instance = model(**kwargs)
        session.add(instance)
        session.commit()
        return instance, True


def remove_empty_values(input_dict):
    """
    Remove entries with empty values from the dictionary.

    Parameters
    ----------
    input_dict (dict): The dictionary from which empty values need to be removed.

    Returns
    -------
    dict: A new dictionary with all empty values removed.

    """
    # Create a new dictionary excluding empty values
    return {key: value for key, value in input_dict.items() if value}
"""Route blueprints.

Blueprints are registered in ``app.app.create_app``. This module intentionally
imports nothing: it used to re-import a subset of the blueprints and expose a
second, unused ``register_routes`` helper that listed four of the six. Two
places disagreeing about which routes the app serves is a bug waiting to
happen, so there is now only one.
"""

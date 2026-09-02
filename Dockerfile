# LifePulse, in the same Python the deploy and CI use.
#
#   docker build -t lifepulse .
#   docker run --rm -p 5000:5000 -e SECRET_KEY="$(python -c 'import secrets;print(secrets.token_hex(32))')" lifepulse
#
# Why this exists at all, given the app runs fine from a virtualenv: the pinned
# scientific stack has to match what built the pickles in app/models/, and
# "works on my machine" is exactly the failure that produced
# "MT19937 is not a known BitGenerator" the last time versions drifted. Render
# reads .python-version, CI reads .python-version, and this reads the same
# number, so all three agree by construction.

FROM python:3.12-slim

# Bytecode written at build time rather than on every cold start, and stdout
# unbuffered so gunicorn's logs reach the platform as they happen rather than
# in blocks -- which matters because app/observability.py puts a request id on
# every line specifically so a user's report can be traced.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Dependencies first, as their own layer: they change far less often than the
# code, so an edit to a template does not reinstall scikit-learn.
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

# Not root. The app writes nothing to disk -- no database, no uploads, no
# session store -- so it has no reason to hold write access to its own code.
RUN useradd --create-home --uid 10001 lifepulse && chown -R lifepulse:lifepulse /app
USER lifepulse

EXPOSE 5000

# The same probe the deploy uses. It reports which models loaded, so a
# container that starts but cannot serve a prediction is unhealthy rather than
# quietly wrong.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:5000/healthz').status == 200 else 1)"

# SECRET_KEY is deliberately not set here. app/app.py refuses to start in
# production without one rather than falling back to a predictable default, and
# baking a key into an image would defeat that on purpose.
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "--access-logfile", "-", "wsgi:app"]

# Centralized Configuration Patterns

A project-agnostic reference for building a configuration subsystem that scales.
Nothing here is specific to this repo — the examples are a generic web service —
so you can carry these patterns to any project, in any language. For how *this*
module applies them, see `README.md` next to this file.

---

## Why configuration is harder than "just make a YAML file"

A naive config ("one YAML everyone edits") feels easy because it silently
collapses **seven independent concerns** into one file. A configuration system
becomes hard — and becomes *worth designing* — precisely because it pulls these
apart and gives each one a clear home:

| # | Concern | The question it answers |
|---|---|---|
| 1 | **Source** | Where do values physically come from? (code defaults, YAML/JSON, `.env`, environment variables, secret managers, remote config servers, CLI flags) |
| 2 | **Precedence** | When two sources set the same key, who wins? |
| 3 | **Schema** | Which keys are valid, of what type, within what range? |
| 4 | **Lifecycle** | When are values resolved — once at startup, per request, or hot-reloaded? |
| 5 | **Secrets** | What must never be committed to version control? |
| 6 | **Environment** | How do dev / staging / prod differ from each other? |
| 7 | **Organization** | Where does a developer go to change a given value? |

The last one has no library that solves it for you — it is pure design, and it
is usually what makes a config system feel intuitive or feel like a swamp.

---

## The core patterns

### Pattern 1 — Separate the data from the loader (the foundation)

The **values** (YAML/JSON/`.env`) are *data* and live in declarative files. The
**code** that locates, reads, validates, merges, and exposes them is *logic* and
lives in a package. Never mix the two: no hardcoded values inside the loader, no
loading logic inside the data files.

```
config/                 # DATA — values only, no logic
  settings.yaml
settings/               # CODE — locates, validates, merges, exposes the values
  schema.py
  runtime.py
  ...
```

**Why:** the two change for different reasons and by different people. Product
and ops tweak *values*; engineers change *loading behavior*. If they share a
file, every value change risks a code change and vice versa. This separation is
what makes every other pattern below possible.

**Seen in:** essentially every mature framework keeps the config file format
separate from the config-loading engine.

### Pattern 2 — Layered override chain (the backbone)

Multiple sources are consulted in a fixed priority order; a higher layer
overrides a lower one for any key it defines. The universal ordering, lowest to
highest priority:

```
code defaults  <  config files  <  environment-specific files  <  env vars  <  explicit/CLI overrides
```

**Why:** it lets one set of committed defaults serve every environment, while
each environment (or each operator, or one test) overrides only what it needs —
without editing shared files.

**Generic example.** Base `settings.yaml` ships `port: 8000`. Production sets an
env var. Nobody edits the YAML:

```yaml
# settings.yaml (committed defaults)
server:
  host: 0.0.0.0
  port: 8000
```

```bash
# production environment
export APP_SERVER__PORT=443     # env var wins over the YAML default
```

**Seen in:** pydantic-settings (`settings_customise_sources`), Spring Boot
`PropertySource` order, .NET `IConfigurationBuilder`, Go's Viper, Python's
Dynaconf. They differ in syntax; the ordering idea is identical.

### Pattern 3 — Typed, validated, fail-fast

Raw strings/dicts are parsed into typed objects and validated **at startup**, so
a bad value crashes immediately with a clear message instead of surfacing as a
mysterious error deep in a request three hours later.

```python
from pydantic import BaseModel

class ServerConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8000          # a non-int in YAML fails HERE, at boot, not later
```

**Why:** configuration errors are the cheapest bugs to fix when they fail loudly
at the boundary, and the most expensive when they leak past it untyped.

**Seen in:** Pydantic, Spring `@ConfigurationProperties` + JSR-303 validation,
JSON-Schema-validated config.

### Pattern 4 — Config is not secrets

Non-sensitive configuration can live in version control; secrets (API keys,
DB passwords, tokens) never can. Keep them on separate paths: config from files,
secrets from environment variables or a secret manager.

```python
class Settings(BaseSettings):
    debug: bool = False                                  # config: fine in YAML/VCS
    database_password: str | None = Field(
        None, alias="DATABASE_PASSWORD"                  # secret: env-only, never YAML
    )
```

**Why:** this is the single most common security mistake in config — an API key
pasted into a committed YAML. Structurally separating the two paths makes the
mistake hard to make.

**Seen in:** the Twelve-Factor App (Factor III), Vault / AWS Secrets Manager /
k8s Secrets integrations.

### Pattern 5 — Environment profiles (base + overlay)

Instead of copy-pasting a whole config per environment, keep one **base** file
plus a small **overlay** per environment, merged by an environment selector.

```
config/
  settings.yaml          # base: shared defaults
  settings.dev.yaml      # overlay: only what dev changes
  settings.prod.yaml     # overlay: only what prod changes
```

```bash
export APP_ENV=prod      # selects settings.prod.yaml to layer on top of the base
```

**Why:** the overlay contains only the *differences*, so environments can't
silently drift apart on values they're supposed to share.

**Seen in:** Spring `application-{profile}.yml`, Rails `config/environments/`,
Dynaconf environments.

> Add this only when you actually run multiple environments. A single `environment`
> field with no overlay files is fine until then — overlays are speculative
> complexity before that point.

### Pattern 6 — Resolve once, share one immutable snapshot

Build the fully-resolved config object **one time** at startup and hand that same
instance to the whole application through a single accessor. Don't re-read files
on every lookup, and don't let different parts of the app resolve config
independently (they'd drift).

```python
from functools import lru_cache

@lru_cache                       # builds once per process, then reuses
def get_settings() -> Settings:
    return Settings()
```

**Why:** cheap access, thread-safe reads, and a guarantee that every module sees
the *same* configuration for the life of the process. In frameworks with
dependency injection, the DI container's singleton scope plays this role instead.

### Pattern 7 — Domain sectioning

Group keys into cohesive, named sections by the area they configure, rather than
one flat bag of a hundred keys.

```python
class Settings(BaseSettings):
    server:   ServerConfig                 # everything about the server
    database: DatabaseConfig               # everything about the database
    logging:  LoggingConfig                # everything about logging
```

**Why:** it makes concern #7 (organization) answerable. "Where do I change the
port?" → the `server` section. A flat namespace forces a grep every time.

### Pattern 8 — Source abstraction

Put each source (file, env, secret manager, remote server) behind a uniform
interface, so you can add, remove, or reorder sources without touching the code
that *reads* config. This is what makes Pattern 2's ordering a one-line change
and lets you bolt on a remote config server later.

**Why:** the set of sources grows over a project's life (you start with YAML +
env, later add Vault or a config service). If consumers depend on the *resolved
object* and not on where values came from, that growth costs nothing downstream.

### Pattern 9 — Break the bootstrap dependency cycle

On the configuration bootstrap path, do **not** depend on components that
themselves need configuration. The classic trap: your application's fancy logger
is *configured from settings* — so if the settings loader logs through that
logger, you get a circular dependency at startup (**settings need the logger, the
logger needs settings**).

Break the cycle at the lowest layer: the file-reading/bootstrap code uses only
primitives that require no configuration.

```python
# low-level config reader — sits on the bootstrap path
import logging                        # stdlib: needs no application config
logger = logging.getLogger(__name__)  # NOT the app logger, which is configured FROM settings

def read_config_file(path): ...
```

**Why:** circular init dependencies are order-dependent and often blow up only in
certain environments (a different import order, a fresh process). Keeping the
lowest layer dependency-free makes startup deterministic. Most codebases have
this latent cycle and never notice until it bites.

**Seen in:** frameworks that bootstrap with a minimal/no-op logger and swap in the
fully-configured one only after configuration has resolved.

### Pattern 10 — Name the two kinds of config: typed settings vs declarative catalogs

Not all configuration is the same animal, and forcing them through one mechanism
is a common source of confusion. Distinguish them explicitly:

- **Typed application settings** — a *fixed, known* schema (server, database,
  logging). Validated into typed objects. Resolved into one settings object. These
  are the application's own knobs.
- **Declarative catalogs** — *open-ended* registries of domain- or user-authored
  entries (e.g. a folder of YAML files, each describing a plugin, a report, a
  workflow, a job). Merged into a dictionary and consumed as data. The schema is
  the shape of *one entry*, not a fixed set of top-level keys.

```
config/
  settings.yaml          # typed settings     -> get_settings() -> Settings object
  reports/               # declarative catalog -> get_reports()  -> {"daily": {...}, ...}
    daily.yaml
    weekly.yaml
```

They deserve *different handling*: typed settings get a validated model plus a
single accessor (Patterns 3 + 6); catalogs get a directory-merge plus their own
accessor and stay as dictionaries (or typed per entry). Give each its own
accessor and say, in the package's front door, which is which.

**Why:** a newcomer needs to know *which mental model applies* before reading a
line. Cramming an open-ended catalog into the fixed settings schema — or
validating a settings section as if it were a loose catalog — fights both.

**Seen in:** application settings (Spring `@ConfigurationProperties`) living
alongside plugin/registry directories (a `plugins/` folder where each plugin
ships its own descriptor file).

---

## A minimal reference implementation (generic)

The patterns above, assembled into one small, complete, project-agnostic
skeleton you can copy as a starting point:

```python
from functools import lru_cache
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Pattern 7: domain sections, each a typed model (Pattern 3)
class ServerConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8000

class DatabaseConfig(BaseModel):
    url: str = "sqlite:///app.db"
    pool_size: int = 5

class Settings(BaseSettings):
    debug: bool = False
    server: ServerConfig = Field(default_factory=ServerConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)

    # Pattern 4: secret comes from the environment, never a committed file
    api_key: str | None = Field(None, alias="API_KEY")

    model_config = SettingsConfigDict(
        env_prefix="APP_",            # APP_SERVER__PORT -> server.port
        env_nested_delimiter="__",    # "__" walks into nested sections
        env_file=".env",             # Pattern 2: one of the layered sources
    )

# Pattern 6: resolve once, share the same snapshot everywhere
@lru_cache
def get_settings() -> Settings:
    return Settings()
```

Precedence (Pattern 2) is whatever the library's default source order is, or an
explicit override hook if the library provides one. The rest of the app only
ever calls `get_settings()` — it never knows or cares which layer a value came
from (Pattern 8). Values live in files, this is the loader (Pattern 1).

---

## The test for "is my config module intuitive?"

A configuration package is well-organized when a newcomer can answer
**"where do I change X?"** without reading the loading code. Map each concern to
exactly one place:

| To change… | Go to… |
|---|---|
| a default value | the base config file |
| something for one environment | that environment's overlay file (Pattern 5) |
| something for one machine/deploy | an environment variable or `.env` |
| what values are *allowed* (new key, type, range) | the typed schema (Pattern 3) |
| which source wins | the precedence definition (Pattern 2) |
| where files live | the paths module |

If any row makes a developer grep the codebase to find the answer, that concern
doesn't have a clear home yet.

---

## A checklist for a new project

1. **Separate the values from the loader** (Pattern 1). Declarative files hold
   values; a code package locates, validates, merges, and exposes them.
2. **Define one typed schema** with defaults (Pattern 3), so config is validated
   and fails fast at startup.
3. **Decide the source precedence explicitly** (Pattern 2) and write it down
   where a reader can see the whole order at once — don't let it be accidental.
4. **Route secrets away from committed files** (Pattern 4) from day one.
5. **Expose one cached accessor** (Pattern 6) and forbid ad-hoc re-resolution
   elsewhere.
6. **Section config by domain** (Pattern 7) so "where do I change X" is always
   answerable.
7. **Keep the bootstrap layer dependency-free** (Pattern 9) so config resolution
   can't deadlock against the very components it configures.
8. **Name your two kinds of config** (Pattern 10) — typed settings vs declarative
   catalogs — and give each its own accessor.
9. **Add environment overlays only when you actually have multiple environments**
   (Pattern 5) — not before.
10. **Write the "where do I change X" table** for your own module. If you can't,
    the organization isn't done yet.
```

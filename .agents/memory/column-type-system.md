---
name: Column type system
description: GenerationSpec.ColumnSpec.type is a strict literal enum; semantic intent uses a separate semantic_type field
---

`ColumnSpec.type` accepts only: `integer | float | string | categorical | datetime | boolean`

`ColumnSpec.semantic_type` carries the semantic label: `id | name | email | address | currency | category | date | phone | none`

The generator uses `semantic_type` to pick the right Faker method (via `strategy: "semantic"` + `faker_method` in the `generation` dict).

**Why:** Separating structural type from semantic intent keeps the schema simple and the generator logic clean. Pydantic enforces the literal at parse time, so sending `type: "email"` raises a 400 immediately.

**How to apply:** When building specs programmatically (in tests, the creator parser, or the profiler), always set `type` to the base structural type and use `semantic_type` for intent.

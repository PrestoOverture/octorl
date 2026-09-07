"""Tests for config_parser."""
import pytest

from config_parser import (
    ConfigError,
    _is_identifier,
    coerce,
    delete_path,
    diff,
    flatten,
    get_path,
    interpolate,
    merge,
    parse_env,
    select,
    set_path,
    to_env,
    unflatten,
    validate,
)


# -- _is_identifier ----------------------------------------------------------


class TestIsIdentifier:
    def test_valid_simple(self):
        assert _is_identifier("foo") is True

    def test_valid_underscore_start(self):
        assert _is_identifier("_bar") is True

    def test_valid_mixed(self):
        assert _is_identifier("A1_b2") is True

    def test_empty(self):
        assert _is_identifier("") is False

    def test_starts_with_digit(self):
        assert _is_identifier("1abc") is False

    def test_contains_dot(self):
        assert _is_identifier("a.b") is False

    def test_contains_hyphen(self):
        assert _is_identifier("a-b") is False


# -- get_path / set_path / delete_path ----------------------------------------


class TestGetPath:
    def test_simple(self):
        assert get_path({"a": 1}, "a") == 1

    def test_nested(self):
        assert get_path({"a": {"b": {"c": 3}}}, "a.b.c") == 3

    def test_default_returned(self):
        assert get_path({"a": 1}, "missing", "default") == "default"

    def test_missing_raises(self):
        with pytest.raises(KeyError):
            get_path({"a": 1}, "no_such_key")

    def test_non_dict_intermediate(self):
        assert get_path({"a": 42}, "a.b", "fallback") == "fallback"

    def test_empty_path_raises(self):
        with pytest.raises(ConfigError, match="empty path"):
            get_path({"a": 1}, "")


class TestSetPath:
    def test_simple(self):
        d: dict = {}
        set_path(d, "key", 10)
        assert d == {"key": 10}

    def test_nested_creates_intermediates(self):
        d: dict = {}
        set_path(d, "a.b.c", 99)
        assert d == {"a": {"b": {"c": 99}}}

    def test_invalid_path_raises(self):
        with pytest.raises(ConfigError, match="invalid path"):
            set_path({}, "a.1bad", 1)

    def test_overwrite_non_dict_intermediate(self):
        d = {"a": "scalar"}
        set_path(d, "a.b", 2)
        assert d == {"a": {"b": 2}}


class TestDeletePath:
    def test_simple(self):
        d = {"a": 1, "b": 2}
        val = delete_path(d, "a")
        assert val == 1
        assert d == {"b": 2}

    def test_nested(self):
        d = {"x": {"y": 10, "z": 20}}
        val = delete_path(d, "x.y")
        assert val == 10
        assert d == {"x": {"z": 20}}

    def test_missing_raises(self):
        with pytest.raises(KeyError):
            delete_path({"a": 1}, "b")

    def test_missing_intermediate_raises(self):
        with pytest.raises(KeyError):
            delete_path({"a": 1}, "a.b.c")

    def test_returns_deleted_value(self):
        d = {"root": {"child": [1, 2, 3]}}
        val = delete_path(d, "root.child")
        assert val == [1, 2, 3]


# -- flatten / unflatten ------------------------------------------------------


class TestFlatten:
    def test_simple(self):
        assert flatten({"a": 1, "b": 2}) == {"a": 1, "b": 2}

    def test_nested(self):
        assert flatten({"a": {"b": 1}, "c": 2}) == {"a.b": 1, "c": 2}

    def test_deeply_nested(self):
        assert flatten({"a": {"b": {"c": 3}}}) == {"a.b.c": 3}

    def test_empty(self):
        assert flatten({}) == {}


class TestUnflatten:
    def test_simple(self):
        assert unflatten({"a.b": 1, "c": 2}) == {"a": {"b": 1}, "c": 2}

    def test_roundtrip(self):
        data = {"x": {"y": {"z": 1}}, "a": 2, "b": {"c": 3}}
        assert unflatten(flatten(data)) == data

    def test_empty(self):
        assert unflatten({}) == {}


# -- merge / diff -------------------------------------------------------------


class TestMerge:
    def test_basic(self):
        assert merge({"a": 1}, {"b": 2}) == {"a": 1, "b": 2}

    def test_override_scalar(self):
        assert merge({"a": 1}, {"a": 99}) == {"a": 99}

    def test_deep_merge(self):
        base = {"db": {"host": "localhost", "port": 5432}}
        over = {"db": {"port": 3306, "name": "mydb"}}
        expected = {"db": {"host": "localhost", "port": 3306, "name": "mydb"}}
        assert merge(base, over) == expected

    def test_does_not_mutate(self):
        base = {"a": {"x": 1}}
        over = {"a": {"y": 2}}
        merge(base, over)
        assert base == {"a": {"x": 1}}


class TestDiff:
    def test_no_changes(self):
        assert diff({"a": 1}, {"a": 1}) == {}

    def test_added_key(self):
        result = diff({}, {"a": 1})
        assert result == {"a": (None, 1)}

    def test_removed_key(self):
        result = diff({"a": 1}, {})
        assert result == {"a": (1, None)}

    def test_changed_value(self):
        result = diff({"a": 1}, {"a": 2})
        assert result == {"a": (1, 2)}

    def test_tuple_order_is_old_new(self):
        """First element is old, second is new."""
        result = diff({"k": "old_value"}, {"k": "new_value"})
        old_val, new_val = result["k"]
        assert old_val == "old_value"
        assert new_val == "new_value"

    def test_nested_diff(self):
        old = {"db": {"port": 5432}}
        new = {"db": {"port": 3306}}
        result = diff(old, new)
        assert result == {"db.port": (5432, 3306)}


# -- validate -----------------------------------------------------------------


class TestValidate:
    def test_passes(self):
        data = {"name": "foo", "count": 5}
        schema = {"name": "str", "count": ["int", "positive"]}
        assert validate(data, schema) == []

    def test_missing_key(self):
        errors = validate({}, {"name": "str"})
        assert len(errors) == 1
        assert "missing" in errors[0]

    def test_failed_rule(self):
        errors = validate({"count": -1}, {"count": "positive"})
        assert any("positive" in e for e in errors)

    def test_oneof_pass(self):
        errors = validate({"mode": "a"}, {"mode": "oneof:a,b,c"})
        assert errors == []

    def test_oneof_fail(self):
        errors = validate({"mode": "bad"}, {"mode": "oneof:a,b,c"})
        assert len(errors) == 1

    def test_bool_not_int(self):
        errors = validate({"flag": True}, {"flag": "int"})
        assert len(errors) == 1

    def test_required_no_op(self):
        assert validate({"a": 1}, {"a": "required"}) == []

    def test_unknown_rule(self):
        errors = validate({"a": 1}, {"a": "nonexistent_rule"})
        assert any("unknown rule" in e for e in errors)

    def test_nonempty_pass(self):
        assert validate({"s": "hi"}, {"s": "nonempty"}) == []

    def test_nonempty_fail(self):
        errors = validate({"s": ""}, {"s": "nonempty"})
        assert len(errors) == 1


# -- parse_env / to_env -------------------------------------------------------


class TestParseEnv:
    def test_basic(self):
        assert parse_env("FOO=bar\n") == {"FOO": "bar"}

    def test_comments_and_blanks(self):
        text = "# a comment\n\nKEY=val\n"
        assert parse_env(text) == {"KEY": "val"}

    def test_double_quoted(self):
        assert parse_env('A="hello world"\n') == {"A": "hello world"}

    def test_single_quoted(self):
        assert parse_env("B='quoted'\n") == {"B": "quoted"}

    def test_missing_equals_raises(self):
        with pytest.raises(ConfigError, match="missing '='"):
            parse_env("NOEQUALSSIGN\n")

    def test_invalid_key_raises(self):
        with pytest.raises(ConfigError, match="invalid key"):
            parse_env("123BAD=val\n")


class TestToEnv:
    def test_basic(self):
        result = to_env({"host": "localhost"})
        assert result.strip() == "HOST=localhost"

    def test_nested_keys(self):
        result = to_env({"db": {"host": "localhost", "port": 5432}})
        assert "DB_HOST=localhost" in result
        assert "DB_PORT=5432" in result

    def test_bool_encoding(self):
        assert "DEBUG=true" in to_env({"debug": True})
        assert "VERBOSE=false" in to_env({"verbose": False})

    def test_sorted_ascending(self):
        result = to_env({"z_key": 1, "a_key": 2})
        lines = result.strip().split("\n")
        assert lines[0].startswith("A_KEY=")
        assert lines[1].startswith("Z_KEY=")

    def test_quoted_spaces(self):
        result = to_env({"msg": "hello world"})
        assert 'MSG="hello world"' in result

    def test_empty_dict(self):
        assert to_env({}) == ""


# -- interpolate --------------------------------------------------------------


class TestInterpolate:
    def test_basic_substitution(self):
        data = {"greeting": "hello ${name}", "name": "world"}
        result = interpolate(data)
        assert result["greeting"] == "hello world"

    def test_external_context(self):
        data = {"msg": "hi ${who}"}
        ctx = {"who": "there"}
        result = interpolate(data, ctx)
        assert result["msg"] == "hi there"

    def test_missing_ref_kept(self):
        data = {"msg": "hi ${unknown}"}
        result = interpolate(data)
        assert result["msg"] == "hi ${unknown}"

    def test_chained_resolution(self):
        data = {"a": "${b}", "b": "${c}", "c": "end"}
        result = interpolate(data)
        assert result["a"] == "end"

    def test_circular_raises(self):
        data = {"a": "${b}", "b": "${a}"}
        with pytest.raises(ConfigError, match="circular"):
            interpolate(data)

    def test_non_string_passthrough(self):
        data = {"num": 42, "label": "val=${num}"}
        result = interpolate(data)
        assert result["num"] == 42
        assert result["label"] == "val=42"

    def test_deep_chain_exceeds_depth_limit(self):
        """A chain of 11 substitutions should exceed the depth-10 limit."""
        data = {f"v{i}": f"${{v{i + 1}}}" for i in range(11)}
        data["v11"] = "end"
        with pytest.raises(ConfigError, match="circular"):
            interpolate(data)


# -- select / coerce ----------------------------------------------------------


class TestSelect:
    def test_picks_paths(self):
        data = {"a": 1, "b": 2, "c": 3}
        assert select(data, ["a", "c"]) == {"a": 1, "c": 3}

    def test_skips_missing(self):
        data = {"a": 1}
        assert select(data, ["a", "nope"]) == {"a": 1}

    def test_nested_path(self):
        data = {"x": {"y": 10, "z": 20}}
        result = select(data, ["x.y"])
        assert result == {"x": {"y": 10}}

    def test_empty_paths(self):
        assert select({"a": 1}, []) == {}


class TestCoerce:
    def test_int(self):
        assert coerce("42") == 42
        assert isinstance(coerce("42"), int)

    def test_negative_int(self):
        assert coerce("-7") == -7

    def test_float(self):
        assert coerce("3.14") == 3.14
        assert isinstance(coerce("3.14"), float)

    def test_bool_true_variants(self):
        for v in ("true", "True", "TRUE", "yes", "Yes", "on", "ON"):
            assert coerce(v) is True

    def test_bool_false_variants(self):
        for v in ("false", "False", "FALSE", "no", "No", "off", "OFF"):
            assert coerce(v) is False

    def test_plain_string(self):
        assert coerce("hello") == "hello"

    def test_empty_string(self):
        assert coerce("") == ""

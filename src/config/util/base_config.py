from __future__ import annotations

from typing import Any, Dict
import argparse
import logging
logger = logging.getLogger(__name__)

_AUTO_DISCOVERY_IGNORE=True

class _Arg:

    def __init__(self, *args, **kwargs) -> None:
        self.args = args
        self.kwargs = kwargs

    def add_argument(self, name : str, parser : argparse.ArgumentParser, prefix : str) -> None:
        arg_name = name
        if prefix is not None and prefix != "":
            arg_name = f"{prefix}.{arg_name}"
        parser.add_argument(f"--{arg_name}", *self.args, **self.kwargs)
        logger.debug(f"Adding argument '{arg_name}' to parser.")

class _BaseConfig:
    _ARG_PREFIX="_arg_"

    def __init__(self) -> None:
        pass

    def _get_arg_name(self, arg : str) -> str:
        return arg.removeprefix(self._ARG_PREFIX)

    def _extend_prefix(self, prefix : str, name : str) -> str:
        if prefix is None or prefix == "":
            return name
        return f"{prefix}.{name}"

    def _full_arg_name(self, prefix : str, name : str) -> str:
        if prefix is None or prefix == "":
            return name
        return f"{prefix}.{name}"

    def _get_args(self) -> Dict[str, _Arg]:
        args = {}
        for attr in self.__dict__.keys():
            if attr.startswith(self._ARG_PREFIX):
                val = getattr(self, attr)
                if not isinstance(val, _Arg):
                    raise Exception(f"{self.__class__.__name__} Expected {attr} to have type {_Arg.__name__} but found {val.__class__.__name__}")
                args[self._get_arg_name(attr)] = val
        return args

    def _get_subconfigs(self) -> Dict[str, _BaseConfig]:
        subconfigs = {}
        for attr in self.__dict__.keys():
            val = getattr(self, attr)
            if isinstance(val, _BaseConfig):
                subconfigs[attr] = val
        return subconfigs

    def add_arguments(self, parser : argparse.ArgumentParser, prefix : str = "") -> None:
        for arg_name, arg in self._get_args().items():
            arg.add_argument(arg_name, parser, prefix)
        for name, subconfig in self._get_subconfigs().items():
            subconfig.add_arguments(parser, self._extend_prefix(prefix, name))

    def parse_arguments(self, args : argparse.Namespace, prefix : str = "") -> None:
        for arg_name in self._get_args().keys():
            setattr(self, arg_name, getattr(args, self._full_arg_name(prefix, arg_name)))
        for name, subconfig in self._get_subconfigs().items():
            subconfig.parse_arguments(args, self._extend_prefix(prefix, name))

    def _get_all(self, prefix : str = "", recursive : bool = True) -> Dict[str, Any]:
        all = {}
        for arg_name in self._get_args().keys():
            all[self._full_arg_name(prefix, arg_name)] = getattr(self, arg_name)
        if recursive:
            for name, subconfig in self._get_subconfigs().items():
                all.update(subconfig._get_all(self._extend_prefix(prefix, name), recursive))
        return all

    def __str__(self) -> str:
        return "\n".join([f"{name}={val}" for name, val in self._get_all().items()])


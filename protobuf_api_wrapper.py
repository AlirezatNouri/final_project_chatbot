"""
Wrapper for Google Protocol Buffer API definitions.

This module provides a convenient interface to create, manipulate, and serialize
Protocol Buffer API definitions using the google.protobuf.api_pb2 module.
"""

from typing import List, Optional
from google.protobuf.api_pb2 import Api, Method, Mixin
from google.protobuf.source_context_pb2 import SourceContext
from google.protobuf.type_pb2 import Syntax, Option


class ProtobufApiWrapper:
    """Wrapper class for working with Protocol Buffer API definitions."""

    @staticmethod
    def create_method(
        name: str,
        request_type_url: str,
        response_type_url: str,
        request_streaming: bool = False,
        response_streaming: bool = False,
        options: Optional[List[Option]] = None,
        syntax: Syntax = Syntax.SYNTAX_PROTO3,
        edition: str = "",
    ) -> Method:
        """
        Create a Protocol Buffer Method definition.

        Args:
            name: The method name
            request_type_url: URL of the request message type
            response_type_url: URL of the response message type
            request_streaming: Whether the request is a stream
            response_streaming: Whether the response is a stream
            options: Optional list of method options
            syntax: Protocol buffer syntax version
            edition: Protocol buffer edition

        Returns:
            Method: A configured Method protobuf message
        """
        method = Method()
        method.name = name
        method.request_type_url = request_type_url
        method.response_type_url = response_type_url
        method.request_streaming = request_streaming
        method.response_streaming = response_streaming
        method.syntax = syntax
        method.edition = edition

        if options:
            method.options.extend(options)

        return method

    @staticmethod
    def create_mixin(name: str, root: str = "") -> Mixin:
        """
        Create a Protocol Buffer Mixin definition.

        Args:
            name: The mixin name
            root: Optional root field name for the mixin

        Returns:
            Mixin: A configured Mixin protobuf message
        """
        mixin = Mixin()
        mixin.name = name
        mixin.root = root
        return mixin

    @staticmethod
    def create_api(
        name: str,
        methods: Optional[List[Method]] = None,
        version: str = "",
        source_context: Optional[SourceContext] = None,
        mixins: Optional[List[Mixin]] = None,
        options: Optional[List[Option]] = None,
        syntax: Syntax = Syntax.SYNTAX_PROTO3,
        edition: str = "",
    ) -> Api:
        """
        Create a Protocol Buffer API definition.

        Args:
            name: The API name
            methods: List of Method definitions
            version: API version string
            source_context: Source context information
            mixins: List of Mixin definitions
            options: List of API options
            syntax: Protocol buffer syntax version
            edition: Protocol buffer edition

        Returns:
            Api: A configured Api protobuf message
        """
        api = Api()
        api.name = name
        api.version = version
        api.syntax = syntax
        api.edition = edition

        if methods:
            api.methods.extend(methods)
        if mixins:
            api.mixins.extend(mixins)
        if options:
            api.options.extend(options)
        if source_context:
            api.source_context.CopyFrom(source_context)

        return api

    @staticmethod
    def serialize_api(api: Api) -> bytes:
        """
        Serialize an API definition to bytes.

        Args:
            api: The Api message to serialize

        Returns:
            bytes: Serialized protobuf message
        """
        return api.SerializeToString()

    @staticmethod
    def deserialize_api(data: bytes) -> Api:
        """
        Deserialize an API definition from bytes.

        Args:
            data: Serialized protobuf message bytes

        Returns:
            Api: Deserialized Api message
        """
        api = Api()
        api.ParseFromString(data)
        return api

    @staticmethod
    def api_to_dict(api: Api) -> dict:
        """
        Convert an API definition to a dictionary.

        Args:
            api: The Api message to convert

        Returns:
            dict: Dictionary representation of the API
        """
        return {
            "name": api.name,
            "version": api.version,
            "syntax": Syntax.Name(api.syntax),
            "edition": api.edition,
            "methods": [
                {
                    "name": method.name,
                    "request_type_url": method.request_type_url,
                    "response_type_url": method.response_type_url,
                    "request_streaming": method.request_streaming,
                    "response_streaming": method.response_streaming,
                }
                for method in api.methods
            ],
            "mixins": [
                {
                    "name": mixin.name,
                    "root": mixin.root,
                }
                for mixin in api.mixins
            ],
            "source_context": {
                "file_name": api.source_context.file_name,
            } if api.source_context else None,
        }

    @staticmethod
    def print_api(api: Api) -> None:
        """
        Pretty print an API definition.

        Args:
            api: The Api message to print
        """
        print(f"API: {api.name}")
        print(f"  Version: {api.version}")
        print(f"  Syntax: {Syntax.Name(api.syntax)}")
        if api.edition:
            print(f"  Edition: {api.edition}")

        if api.methods:
            print("  Methods:")
            for method in api.methods:
                print(f"    - {method.name}")
                print(f"      Request: {method.request_type_url}")
                print(f"      Response: {method.response_type_url}")
                if method.request_streaming:
                    print(f"      Request Streaming: true")
                if method.response_streaming:
                    print(f"      Response Streaming: true")

        if api.mixins:
            print("  Mixins:")
            for mixin in api.mixins:
                print(f"    - {mixin.name}")
                if mixin.root:
                    print(f"      Root: {mixin.root}")


# Convenience functions for direct use
def create_api(
    name: str,
    methods: Optional[List[Method]] = None,
    version: str = "",
    source_context: Optional[SourceContext] = None,
    mixins: Optional[List[Mixin]] = None,
    options: Optional[List[Option]] = None,
) -> Api:
    """Convenience function to create an API definition."""
    return ProtobufApiWrapper.create_api(
        name, methods, version, source_context, mixins, options
    )


def create_method(
    name: str,
    request_type_url: str,
    response_type_url: str,
    request_streaming: bool = False,
    response_streaming: bool = False,
) -> Method:
    """Convenience function to create a Method definition."""
    return ProtobufApiWrapper.create_method(
        name, request_type_url, response_type_url, request_streaming, response_streaming
    )


def create_mixin(name: str, root: str = "") -> Mixin:
    """Convenience function to create a Mixin definition."""
    return ProtobufApiWrapper.create_mixin(name, root)

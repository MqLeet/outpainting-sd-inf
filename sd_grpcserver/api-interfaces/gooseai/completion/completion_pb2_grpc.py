# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

import completion_pb2 as completion__pb2


class CompletionServiceStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.Completion = channel.unary_stream(
                '/gooseai.CompletionService/Completion',
                request_serializer=completion__pb2.Request.SerializeToString,
                response_deserializer=completion__pb2.Answer.FromString,
                )


class CompletionServiceServicer(object):
    """Missing associated documentation comment in .proto file."""

    def Completion(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_CompletionServiceServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'Completion': grpc.unary_stream_rpc_method_handler(
                    servicer.Completion,
                    request_deserializer=completion__pb2.Request.FromString,
                    response_serializer=completion__pb2.Answer.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'gooseai.CompletionService', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class CompletionService(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def Completion(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_stream(request, target, '/gooseai.CompletionService/Completion',
            completion__pb2.Request.SerializeToString,
            completion__pb2.Answer.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

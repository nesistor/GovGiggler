import 'package:flutter/material.dart';
import 'dart:convert';
import 'package:http/http.dart' as http;
import 'dart:io';
import 'package:http_parser/http_parser.dart';

import '../models/models.dart';

class ApiProvider with ChangeNotifier {
  final List<Message> _messages = [];

  List<Message> get messages => _messages;

  Future<void> generateResponse(String question) async {
    var url = Uri.parse('http://127.0.0.1:8000/generate-response');
    try {
      _addMessage(Message(message: question, isUserMessage: true));

      var response = await http.post(
        url,
        headers: {'Content-Type': 'application/json; charset=utf-8'},
        body: json.encode({'question': question}),
      );

      if (response.statusCode == 200) {
        var responseBody = utf8.decode(response.bodyBytes);
        var responseData = json.decode(responseBody);

        if (responseData is List && responseData.isNotEmpty) {
          String serverResponse = responseData[0];
          _addMessage(Message(
            message: serverResponse,
            isUserMessage: false,
            isMarkdown: true,
          ));
        }
      } else {
        throw Exception('Failed to load response');
      }
    } catch (e) {
      _addMessage(Message(
        message: 'Error: $e',
        isUserMessage: false,
      ));
    }
  }

  Future<void> uploadDocument(String filePath) async {
    var url = Uri.parse('http://127.0.0.1:8000/validate-document');
    try {
      _addMessage(Message(
        message: 'Uploading document...',
        isUserMessage: true,
      ));

      String mimeType = _detectMimeType(filePath);

      var request = http.MultipartRequest('POST', url);
      request.files.add(await http.MultipartFile.fromPath(
        'file',
        filePath,
        contentType: MediaType.parse(mimeType),
      ));

      var streamedResponse = await request.send();
      var response = await http.Response.fromStream(streamedResponse);

      if (response.statusCode == 200) {
        var responseBody = utf8.decode(response.bodyBytes);
        var responseData = json.decode(responseBody);

        String resultMessage = responseData['content'] ?? 'No content available';
        _addMessage(Message(
          message: resultMessage,
          isUserMessage: false,
          isMarkdown: true,
        ));
      } else if (response.statusCode == 400) {
        var responseBody = utf8.decode(response.bodyBytes);
        var responseData = json.decode(responseBody);

        _addMessage(Message(
          message: responseData['detail'] ?? 'Unsupported file type.',
          isUserMessage: false,
        ));
      } else {
        _addMessage(Message(
          message: 'Document upload failed: ${response.reasonPhrase}',
          isUserMessage: false,
        ));
      }
    } catch (e) {
      _addMessage(Message(
        message: 'Error occurred: $e',
        isUserMessage: false,
      ));
    }
  }

  void _addMessage(Message message) {
    _messages.add(message);
    notifyListeners();
  }

  String _detectMimeType(String filePath) {
    if (filePath.endsWith('.pdf')) {
      return 'application/pdf';
    } else if (filePath.endsWith('.docx')) {
      return 'application/vnd.openxmlformats-officedocument.wordprocessingml.document';
    } else {
      throw Exception('Unsupported file type. Only PDF and DOCX are allowed.');
    }
  }
}

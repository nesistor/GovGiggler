import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../providers/api_provider.dart';
import '../widgets/chat_bubble.dart';
import 'package:file_picker/file_picker.dart';

class ChatPage extends StatelessWidget {
  const ChatPage({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Government Assistant'), // Updated title
        backgroundColor: Colors.black,
        centerTitle: true,
      ),
      body: const ChatBody(),
    );
  }
}

class ChatBody extends StatefulWidget {
  const ChatBody({super.key});

  @override
  State<ChatBody> createState() => _ChatBodyState();
}

class _ChatBodyState extends State<ChatBody> {
  final TextEditingController _controller = TextEditingController();

  // Function to send the message
  void _sendMessage() async {
    if (_controller.text.trim().isEmpty) return;

    final apiProvider = Provider.of<ApiProvider>(context, listen: false);
    await apiProvider.generateResponse(_controller.text);

    _controller.clear();
  }

  // Function to handle document upload
  void _addDocument() async {
    final apiProvider = Provider.of<ApiProvider>(context, listen: false);

    try {
      FilePickerResult? result = await FilePicker.platform.pickFiles(
        type: FileType.custom,
        allowedExtensions: ['pdf', 'doc', 'docx'],
      );

      if (result != null) {
        String filePath = result.files.single.path!;
        await apiProvider.uploadDocument(filePath); // Upload the document for validation
      }
    } catch (e) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Failed to pick or upload document: $e')),
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    final apiProvider = Provider.of<ApiProvider>(context);

    return Column(
      children: [
        Expanded(
          child: ListView.builder(
            itemCount: apiProvider.messages.length,
            itemBuilder: (context, index) {
              final message = apiProvider.messages[index];
              return ChatBubble(
                message: message.message,
                isUserMessage: message.isUserMessage,
                isMarkdown: message.isMarkdown,
              );
            },
          ),
        ),
        Padding(
          padding: const EdgeInsets.all(8.0),
          child: Row(
            children: [
              // Add Document Icon Button
              IconButton(
                icon: const Icon(Icons.attach_file, color: Colors.white),
                onPressed: _addDocument,
              ),
              Expanded(
                child: TextField(
                  controller: _controller,
                  style: const TextStyle(color: Colors.white),
                  decoration: const InputDecoration(
                    hintText: 'Type your message...',
                    hintStyle: TextStyle(color: Colors.white),
                    border: OutlineInputBorder(),
                    filled: true,
                    fillColor: Colors.black,
                  ),
                ),
              ),
              IconButton(
                icon: const Icon(Icons.send, color: Colors.white),
                onPressed: _sendMessage,
              ),
            ],
          ),
        ),
      ],
    );
  }
}

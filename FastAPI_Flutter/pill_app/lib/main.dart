import 'package:flutter/foundation.dart';
import 'dart:io';

import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';

import 'dart:convert';

import 'package:http/http.dart' as http;

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: '약품 불량 탐지',
      debugShowCheckedModeBanner: false,
      home: const HomePage(),
    );
  }
}

class HomePage extends StatefulWidget {
  const HomePage({super.key});

  @override
  State<HomePage> createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> {
  XFile? selectedImage;

  String resultText = "";

  bool isBlurWarning = false;

  Future<void> pickImage() async {
    final picker = ImagePicker();

    final image = await picker.pickImage(
      source: ImageSource.gallery,
    );

    if (image != null) {
      setState(() {
        selectedImage = image;
        resultText = "";
        isBlurWarning = false;
      });
    }
  }

Future<void> analyzeImage() async {

  print("1");

  if (selectedImage == null) return;

  print("2");

  var request = http.MultipartRequest(
    "POST",
    Uri.parse(
       "https://static-overarch-ninth.ngrok-free.dev/predict",
    ),
  );

  print("3");

  if (kIsWeb) {

    final bytes =
        await selectedImage!.readAsBytes();

    print("4");

    request.files.add(
      http.MultipartFile.fromBytes(
        "file",
        bytes,
        filename: selectedImage!.name,
      ),
    );

  print("5");

  } else {

    request.files.add(
      await http.MultipartFile.fromPath(
        "file",
        selectedImage!.path,
      ),
    );

  }
  print("6");

  var response =
      await request.send();
  print("7");

  if (response.statusCode == 200) {

    final body =
        await response.stream.bytesToString();
    print("8");

    final data =
        jsonDecode(body);
    print(data);

    if (data["is_blur"] == true) {

      print("BLUR DETECTED");


      setState(() {
        isBlurWarning = true;
        resultText =
"""
사진이 흐립니다.

카메라 렌즈를 닦고 다시 촬영해주세요.

Blur Score : ${data["blur_score"]}
""";

    });

      return;
  }
    setState(() {
      isBlurWarning = false;
      resultText =
"""
약품명 : ${data["drug_name"]}

불량 여부 : ${data["is_defect"]}

신뢰도 : ${data["confidence"]}
""";

    });
    
  } else {

    setState(() {

      resultText =
          "서버 오류 : ${response.statusCode}";

    });

  }
}

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text("약품 불량 탐지"),
      ),
      body: Center(
        child: SingleChildScrollView(
          child: Column(
            mainAxisAlignment:
                MainAxisAlignment.center,
            children: [

              ElevatedButton(
                onPressed: pickImage,
                child: const Text(
                  "사진 선택",
                ),
              ),

              ElevatedButton(
                onPressed: analyzeImage,
                child: const Text(
                  "분석 시작",
                ),
              ),

              const SizedBox(
                height: 20,
              ),

              if (selectedImage != null)
                kIsWeb
                    ? Image.network(
                        selectedImage!.path,
                        height: 300,
                      )
                    : Image.file(
                        File(selectedImage!.path),
                        height: 300,
                      ),

              const SizedBox(
                height: 20,
              ),
              Text(
                resultText,
                textAlign: TextAlign.center,
                style: TextStyle(
                  color:
                      isBlurWarning
                          ? Colors.red
                          : Colors.black,
                  fontSize: 18,
                  fontWeight:
                      isBlurWarning
                          ? FontWeight.bold
                          : FontWeight.normal,
                  ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
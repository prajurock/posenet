package com.example.pose;

import android.Manifest;
import android.app.Activity;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.support.annotation.NonNull;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.Toast;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.Tensor;

public class MainActivity extends AppCompatActivity {
  private static final int CAMERA_REQUEST = 1888;
  private ImageView imageView;
  private static final int MY_CAMERA_PERMISSION_CODE = 100;
  Interpreter tflite = null;
  private String TAG = "Prajwal";

  Map<Integer, Object> outputMap = new HashMap<>();
  float[][][][] out1 = new float[1][22][22][17];
  float[][][][] out2 = new float[1][22][22][34];
  float[][][][] out3 = new float[1][22][22][32];
  float[][][][] out4 = new float[1][22][22][32];


  @Override
  protected void onCreate(Bundle savedInstanceState) {
    super.onCreate(savedInstanceState);
    setContentView(R.layout.activity_main);
    String modelFile="posenet_mv1_075_float_from_checkpoints.tflite";
    try {
      tflite=new Interpreter(loadModelFile(MainActivity.this,modelFile));
    } catch (IOException e) {
      e.printStackTrace();
    }
    final Tensor no = tflite.getInputTensor(0);
    Log.d(TAG, "onCreate: Input shape"+ Arrays.toString(no.shape()));

    int c = tflite.getOutputTensorCount();
    Log.d(TAG, "onCreate: Output Count" +c );
    for (int i = 0; i <4 ; i++) {
      final Tensor output = tflite.getOutputTensor(i);
      Log.d(TAG, "onCreate: Output shape" + Arrays.toString(output.shape()));
    }
    this.imageView =  this.findViewById(R.id.imageView1);
    Button photoButton = this.findViewById(R.id.button1);
    photoButton.setOnClickListener(new View.OnClickListener() {

      @Override
      public void onClick(View v) {
        if (checkSelfPermission(Manifest.permission.CAMERA)
            != PackageManager.PERMISSION_GRANTED) {
          requestPermissions(new String[]{Manifest.permission.CAMERA},
              MY_CAMERA_PERMISSION_CODE);
        } else {
          Intent cameraIntent = new Intent(android.provider.MediaStore.ACTION_IMAGE_CAPTURE);
          startActivityForResult(cameraIntent, CAMERA_REQUEST);
        }
      }
    });
  }

  public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull
      int[] grantResults) {
    super.onRequestPermissionsResult(requestCode, permissions, grantResults);
    if (requestCode == MY_CAMERA_PERMISSION_CODE) {
      if (grantResults[0] == PackageManager.PERMISSION_GRANTED) {
        Toast.makeText(this, "camera permission granted", Toast.LENGTH_LONG).show();
        Intent cameraIntent = new
            Intent(android.provider.MediaStore.ACTION_IMAGE_CAPTURE);
        startActivityForResult(cameraIntent, CAMERA_REQUEST);
      } else {
        Toast.makeText(this, "camera permission denied", Toast.LENGTH_LONG).show();
      }
    }
  }

  protected void onActivityResult ( int requestCode, int resultCode, Intent data){
    if (requestCode == CAMERA_REQUEST && resultCode == Activity.RESULT_OK) {
      Bitmap photo = (Bitmap) data.getExtras().get("data");
      Log.d(TAG,"bhai:"+photo.getWidth()+":"+photo.getHeight());
      //imageView.setImageBitmap(photo);
      photo = Bitmap.createScaledBitmap(photo, 337, 337, false);
      photo = photo.copy(Bitmap.Config.ARGB_8888,true);
      Log.d(TAG, "onActivityResult: Bitmap resized");

      int width =photo.getWidth();
      int height = photo.getHeight();
      float[][][][] result = new float[1][width][height][3];
      int[] pixels = new int[width*height];
      photo.getPixels(pixels, 0, width, 0, 0, width, height);
      int pixelsIndex = 0;
      for (int i = 0; i < width; i++)
      {
        for (int j = 0; j < height; j++)
        {
          // result[i][j] =  pixels[pixelsIndex];
          int p = pixels[pixelsIndex];
          result[0][i][j][0]  = (p >> 16) & 0xff;
          result[0][i][j][1]  = (p >> 8) & 0xff;
          result[0][i][j][2]  = p & 0xff;
          pixelsIndex++;
        }
      }
      Object [] inputs = {result};
      //inputs[0] = inp;

      outputMap.put(0, out1);
      outputMap.put(1, out2);
      outputMap.put(2, out3);
      outputMap.put(3, out4);

      tflite.runForMultipleInputsOutputs(inputs,outputMap);
      out1 = (float[][][][]) outputMap.get(0);
      out2 = (float[][][][]) outputMap.get(1);
      out3 = (float[][][][]) outputMap.get(2);
      out4 = (float[][][][]) outputMap.get(3);

      Canvas canvas = new Canvas(photo);
      Paint p = new Paint();
      p.setColor(Color.RED);

      float[][][] scores = new float[out1[0].length][out1[0][0].length][17];
      int[][] heatmap_pos = new int[17][2];

      for(int i=0;i<17;i++)
      {
        float max = -1;

        for(int j=0;j<out1[0].length;j++)
        {
          for(int k=0;k<out1[0][0].length;k++)
          {
            //  Log.d("mylog", "onActivityResult: "+out1[0][j][k][i]);
            scores[j][k][i]  = sigmoid(out1[0][j][k][i]);
            if(max<scores[j][k][i])
            {
              max = scores[j][k][i];
              heatmap_pos[i][0] = j;
              heatmap_pos[i][1] = k;
            }
          }

        }
            Log.d(TAG, "onActivityResult: "+max+"    "+heatmap_pos[i][0]+"    "+heatmap_pos[i][1]);
      }

      for(int i=0;i<17;i++)
      {
        Log.d("heatlog", "onActivityResult: "+heatmap_pos[i][0]+"    "+heatmap_pos[i][1]);
      }
      float[][] offset_vector = new float[17][2];
      float[][] keypoint_pos = new float[17][2];
      for(int i=0;i<17;i++)
      {
        offset_vector[i][0] = out2[0][heatmap_pos[i][0]][heatmap_pos[i][1]][i];
        offset_vector[i][1] = out2[0][heatmap_pos[i][0]][heatmap_pos[i][1]][i+17];
        Log.d("myoff",offset_vector[i][0]+":"+offset_vector[i][1]);
        keypoint_pos[i][0] = heatmap_pos[i][0]*16+offset_vector[i][0];
        keypoint_pos[i][1] = heatmap_pos[i][1]*16+offset_vector[i][1];
        Log.d(TAG, "onActivityResult: "+keypoint_pos[i][0]+"    "+keypoint_pos[i][1]);
        canvas.drawCircle(keypoint_pos[i][0],keypoint_pos[i][1],5,p);      }

      imageView.setImageBitmap(photo);
    }
  }

  private MappedByteBuffer loadModelFile(Activity activity, String MODEL_FILE) throws IOException {
    AssetFileDescriptor fileDescriptor = activity.getAssets().openFd(MODEL_FILE);
    FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
    FileChannel fileChannel = inputStream.getChannel();
    long startOffset = fileDescriptor.getStartOffset();
    long declaredLength = fileDescriptor.getDeclaredLength();
    return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
  }

  public float sigmoid(float value) {
    float p =  (float)(1.0 / (1 + Math.exp(-value)));
    return p;
  }
}
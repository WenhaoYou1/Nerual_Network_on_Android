package com.example.nom;

import android.Manifest;
import android.annotation.SuppressLint;
import android.app.ActivityManager;
import android.content.Intent;
import android.content.pm.PackageInfo;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.os.Bundle;
import android.os.Environment;
import android.os.Handler;
import android.os.Looper;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.Toast;

import androidx.activity.result.ActivityResult;
import androidx.activity.result.ActivityResultCallback;
import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import com.chaquo.python.PyObject;
import com.chaquo.python.Python;
import com.chaquo.python.android.AndroidPlatform;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.OutputStream;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class MainActivity extends AppCompatActivity {

    byte[] byte_array;
    ExecutorService executor = Executors.newSingleThreadExecutor();
    private static final int REQUEST_WRITE_STORAGE_REQUEST_CODE = 1;
    Bitmap bitmap_result;
    byte[] result;

    ActivityResultLauncher<Intent> activityResultLauncher =
            registerForActivityResult(
                    new ActivityResultContracts.StartActivityForResult(),
                    new ActivityResultCallback<ActivityResult>() {
                        @Override
                        public void onActivityResult(ActivityResult o) {
                            int result = o.getResultCode();
                            Intent data = o.getData();

                            if (result == RESULT_OK) {
                                // Toast.makeText(MainActivity.this, "Passed", Toast.LENGTH_LONG).show();
                                assert data != null;
                                Uri imageUri = data.getData();

                                try {
                                    // decode the image
                                    assert imageUri != null;
                                    Bitmap bitmap = BitmapFactory.decodeStream(getContentResolver().openInputStream(imageUri));
                                    ByteArrayOutputStream stream = new ByteArrayOutputStream();
                                    bitmap.compress(Bitmap.CompressFormat.PNG, 100, stream);
                                    byte_array = stream.toByteArray();


                                    // System.out.println(byte_array);
                                    Toast.makeText(MainActivity.this, "Passed Tensor", Toast.LENGTH_LONG).show();

                                    run();
                                } catch (FileNotFoundException e) {
                                    e.printStackTrace();
                                }
                            }
                            else {
                                Toast.makeText(MainActivity.this, "Failed", Toast.LENGTH_LONG).show();
                            }
                        }
                    }
            );

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        Button cam = (Button) findViewById(R.id.cam);
        Button lib = (Button) findViewById(R.id.lib);
        Button run_b = (Button) findViewById(R.id.run);

        memory();
        deleteAppDirectory("output");

        if (ContextCompat.checkSelfPermission(MainActivity.this, Manifest.permission.WRITE_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(MainActivity.this, new String[]{Manifest.permission.WRITE_EXTERNAL_STORAGE}, REQUEST_WRITE_STORAGE_REQUEST_CODE);
        }


        cam.setOnClickListener(new View.OnClickListener() {
            @SuppressLint("SetTextI18n")
            @Override
            public void onClick(View view) {
            }
        });

        lib.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                folder_creation();
                Intent intent = new Intent(Intent.ACTION_OPEN_DOCUMENT);
                intent.addCategory(Intent.CATEGORY_OPENABLE);
                intent.setType("*/*");
                activityResultLauncher.launch(intent);
            }
        });

        run_b.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                folder_creation();
                run_bulk();
            }
        });
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == REQUEST_WRITE_STORAGE_REQUEST_CODE) {
            if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                // The permission request was granted
                Toast.makeText(MainActivity.this, "Access Granted", Toast.LENGTH_LONG).show();
            } else {
                // The permission request was denied
                // Handle the case where permission is denied
                Toast.makeText(MainActivity.this, "Required Permission to Save Image", Toast.LENGTH_LONG).show();
            }
        }
    }


    void saveImage() {
        File picturesDirectory = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_PICTURES);
        File imageFile = new File(picturesDirectory, "result_import" + ".png");
        try (OutputStream out = new FileOutputStream(imageFile)) {
            bitmap_result.compress(Bitmap.CompressFormat.PNG, 100, out);
            out.flush();
        }
        catch (Exception e) {
            e.printStackTrace();
            Toast.makeText(MainActivity.this, "Error saving image", Toast.LENGTH_SHORT).show();
            return;
        }

        Intent mediaScanIntent = new Intent(Intent.ACTION_MEDIA_SCANNER_SCAN_FILE);
        File f = new File(imageFile.getAbsolutePath());
        Uri contentUri = Uri.fromFile(f);
        mediaScanIntent.setData(contentUri);
        MainActivity.this.sendBroadcast(mediaScanIntent);
    }

    void memory() {
        ActivityManager.MemoryInfo mi = new ActivityManager.MemoryInfo();
        ActivityManager activityManager = (ActivityManager) getSystemService(ACTIVITY_SERVICE);
        activityManager.getMemoryInfo(mi);
        double availableMegs = mi.availMem / 0x100000L;

        double percentAvail = 100 - (mi.availMem / (double) mi.totalMem * 100.0);

        Log.d("available memory", String.format ("%.0f", availableMegs));
        Log.d("percentage used memory", String.format ("%.0f", percentAvail));
    }

    void run_bulk() {
        new Handler(Looper.getMainLooper()).post(new Runnable() {
            @Override
            public void run() {
                Toast.makeText(MainActivity.this, "Running...", Toast.LENGTH_SHORT).show();
            }
        });
        executor.execute(new Runnable() {
            @Override
            public void run() {
                // Run Python script using Chaquopy
                if (!Python.isStarted()) {
                    Log.d("debug", "run3");
                    Python.start(new AndroidPlatform(MainActivity.this));
                }
                Log.d("debug", "run4");
                Python py = Python.getInstance();
                Log.d("debug", "run5");
                PyObject pyobj = py.getModule("detect_UNetFR");
                Log.d("debug", "run6");
//                result = pyobj.callAttr("main", byte_array).toJava(byte[].class);
//                Log.d("debug", "run7");
                // return pyobj.toString();
//                bitmap_result = BitmapFactory.decodeByteArray(result, 0, result.length);

                // Run on the main thread after script is executed
                new Handler(Looper.getMainLooper()).post(new Runnable() {
                    @Override
                    public void run() {
                        // saveImage();
                        // Update UI, for example, showing a Toast
                        Toast.makeText(MainActivity.this, "Done", Toast.LENGTH_SHORT).show();

                        // Continue with the result
                        // continueWithResult(result.toString());
                        // deleteAppDirectory("tensor");
                    }
                });
            }
        });
    }
    void run() {
        // Display a Toast message from the background thread
        new Handler(Looper.getMainLooper()).post(new Runnable() {
            @Override
            public void run() {
                Toast.makeText(MainActivity.this, "Running...", Toast.LENGTH_SHORT).show();
            }
        });
        executor.execute(new Runnable() {
            @Override
            public void run() {
                // Run Python script using Chaquopy
                if (!Python.isStarted()) {
                    Log.d("debug", "run3");
                    Python.start(new AndroidPlatform(MainActivity.this));
                }
                Log.d("debug", "run4");
                Python py = Python.getInstance();
                Log.d("debug", "run5");
                PyObject pyobj = py.getModule("lib_UNetFR");
                Log.d("debug", "run6");
                result = pyobj.callAttr("main", byte_array).toJava(byte[].class);
                Log.d("debug", "run7");
                // return pyobj.toString();
                bitmap_result = BitmapFactory.decodeByteArray(result, 0, result.length);

                // Run on the main thread after script is executed
                new Handler(Looper.getMainLooper()).post(new Runnable() {
                    @Override
                    public void run() {
                        saveImage();
                        // Update UI, for example, showing a Toast
                        Toast.makeText(MainActivity.this, "Done", Toast.LENGTH_SHORT).show();

                        // Continue with the result
                        // continueWithResult(result.toString());
                        deleteAppDirectory("tensor");
                    }
                });
            }
        });
    }

    void folder_creation() {
        PackageManager m = getPackageManager();
        String s = getPackageName();
        try {
            PackageInfo p = m.getPackageInfo(s, 0);
            s = p.applicationInfo.dataDir;

            // Path for the new directory
            String tensor_dir = s + File.separator + "tensor";
            String output_dir = s + File.separator + "output";

            // Create a File object for the new directory
            File t_dir = new File(tensor_dir);
            File o_dir = new File(output_dir);

            // Check if the directory exists. If not, create it.
            if (!t_dir.exists()) {
                if (t_dir.mkdir()) {
                    Log.d("DIR", "Directory created: " + t_dir);
                }
                else {
                    Log.d("DIR", "Failed to create directory: " + t_dir);
                }
            }
            else {
                Log.d("DIR", "Directory already exists: " + t_dir);
            }
            // output dir
            if (!o_dir.exists()) {
                if (o_dir.mkdir()) {
                    Log.d("DIR", "Directory created: " + o_dir);
                }
                else {
                    Log.d("DIR", "Failed to create directory: " + o_dir);
                }
            }
            else {
                Log.d("DIR", "Directory already exists: " + o_dir);
            }
        }
        catch (PackageManager.NameNotFoundException e) {
            Log.d("DIR", "Error: Package Not Found ", e);
        }
    }

    private void deleteAppDirectory(String dirName) {
        PackageManager m = getPackageManager();
        String packageName = getPackageName();
        try {
            PackageInfo p = m.getPackageInfo(packageName, 0);
            String dataDir = p.applicationInfo.dataDir;

            // Path for the directory you want to delete
            String dirPath = dataDir + File.separator + dirName;
            File dirToDelete = new File(dirPath);

            if (deleteDirectory(dirToDelete)) {
                Log.d("DIR", "Directory deleted successfully: " + dirPath);
            } else {
                Log.d("DIR", "Failed to delete directory: " + dirPath);
            }
        } catch (PackageManager.NameNotFoundException e) {
            Log.d("DIR", "Error: Package Not Found ", e);
        }
    }

    private boolean deleteDirectory(File dir) {
        if (dir.isDirectory()) {
            String[] children = dir.list();
            assert children != null;
            for (String child : children) {
                boolean success = deleteDirectory(new File(dir, child));
                if (!success) {
                    return false;  // If failed to delete, return false
                }
            }
        }
        // The directory is now empty or it is a file, so delete it
        return dir.delete();
    }
}
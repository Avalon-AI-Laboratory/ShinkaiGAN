// import { NextResponse } from "next/server";
// import fs from "fs";
// import path from "path";

// export async function GET() {
//   try {
//     const baseDir = path.resolve("./public/images");

//     const getImagesFromDirectory = (directory: string): string[] => {
//       const dirPath = path.join(baseDir, directory);
//       const files = fs.readdirSync(dirPath);
//       return files.map((file) => `/images/${directory}/${file}`);
//     };

//     const animeganImages = getImagesFromDirectory("animegan");
//     const cartoonganImages = getImagesFromDirectory("cartoongan");
//     const scenemifyImages = getImagesFromDirectory("scenemify");
//     const nijiganImages = getImagesFromDirectory("nijigan");

//     const allImages = [
//       ...animeganImages,
//       ...cartoonganImages,
//       ...scenemifyImages,
//       ...nijiganImages,
//     ];

//     // Shuffle and return 10 random images
//     const shuffledImages = allImages
//       .sort(() => 0.5 - Math.random())
//       .slice(0, 10);

//     // Respond with the shuffled images array as JSON
//     return NextResponse.json(shuffledImages);
//   } catch (error) {
//     console.error("Error fetching images:", error);
//     return NextResponse.json(
//       { error: "Failed to load images" },
//       { status: 500 }
//     );
//   }
// }

export const dynamic = "force-dynamic";

import { NextResponse } from "next/server";
import fs from "fs";
import path from "path";

export async function GET() {
  try {
    const baseDir = path.resolve("./src/images");

    // Fungsi untuk mengambil gambar dari direktori tertentu
    const getImagesFromDirectory = (directory: string): string[] => {
      const dirPath = path.join(baseDir, directory);
      const files = fs.readdirSync(dirPath);
      return files.map((file) => `/images/${directory}/${file}`);
    };

    // Mengambil gambar dari masing-masing kategori
    const animeganImages = getImagesFromDirectory("animegan");
    const cartoonganImages = getImagesFromDirectory("cartoongan");
    const scenemifyImages = getImagesFromDirectory("scenemify");
    const nijiganImages = getImagesFromDirectory("nijigan");

    // Gabungkan semua gambar dari setiap kategori
    const allImages = [
      ...animeganImages,
      ...cartoonganImages,
      ...scenemifyImages,
      ...nijiganImages,
    ];

    // Fungsi pengacakan Fisher-Yates
    const shuffleArray = (array: string[]): string[] => {
      for (let i = array.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [array[i], array[j]] = [array[j], array[i]];
      }
      return array;
    };

    // Pengacakan seluruh gambar
    const shuffledImages = shuffleArray(allImages);

    // Pilih 10 gambar acak
    const selectedImages = shuffledImages.slice(0, 10);

    // Respon dengan array gambar yang telah dipilih
    return NextResponse.json(selectedImages, {
      headers: {
        "Cache-Control": "no-store",
      },
    });
  } catch (error) {
    console.error("Error fetching images:", error);
    return NextResponse.json(
      { error: "Failed to load images" },
      { status: 500 }
    );
  }
}

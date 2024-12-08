import { NextResponse } from "next/server";
import fs from "fs";
import path from "path";

export async function GET() {
  try {
    const baseDir = path.resolve("./public/images");

    const getImagesFromDirectory = (directory: string): string[] => {
      const dirPath = path.join(baseDir, directory);
      const files = fs.readdirSync(dirPath);
      return files.map((file) => `/images/${directory}/${file}`);
    };

    const animeganImages = getImagesFromDirectory("animegan");
    const cartoonganImages = getImagesFromDirectory("cartoongan");
    const scenemifyImages = getImagesFromDirectory("scenemify");
    const nijiganImages = getImagesFromDirectory("nijigan");

    const allImages = [
      ...animeganImages,
      ...cartoonganImages,
      ...scenemifyImages,
      ...nijiganImages,
    ];

    // Shuffle and return 10 random images
    const shuffledImages = allImages
      .sort(() => 0.5 - Math.random())
      .slice(0, 10);

    // Respond with the shuffled images array as JSON
    return NextResponse.json(shuffledImages);
  } catch (error) {
    console.error("Error fetching images:", error);
    return NextResponse.json(
      { error: "Failed to load images" },
      { status: 500 }
    );
  }
}

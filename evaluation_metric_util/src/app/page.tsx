"use client";

import { useState, useEffect } from "react";
import { Form } from "@/components/primitive/form";
import { Button } from "@/components/ui/button";
import { useToast } from "@/hooks/use-toast";
import { ToastAction } from "@/components/ui/toast";
import Image from "next/image";

export default function Home() {
  const [imageUrls, setImageUrls] = useState<string[]>([]);
  const [ratings, setRatings] = useState<string[]>([]);
  const [loading, setLoading] = useState(false); // Loading state
  const { toast } = useToast();

  // Fetch random images from the server on component mount
  useEffect(() => {
    const fetchImages = async () => {
      try {
        const response = await fetch("/api/images");

        if (!response.ok) {
          throw new Error("Failed to fetch images");
        }

        const images = await response.json();
        console.log("Fetched images:", images); // Debugging log

        setImageUrls(images);
        setRatings(Array(images.length).fill("1")); // Initialize ratings array based on image length
      } catch (error) {
        console.error("Error fetching images:", error);
      }
    };

    fetchImages();
  }, []);

  // Function to handle the rating change for each image
  const handleRatingChange = (index: number, rating: string) => {
    const updatedRatings = [...ratings];
    updatedRatings[index] = rating;
    setRatings(updatedRatings);
  };

  // Function to handle the submission of the form
  const handleFinish = async () => {
    setLoading(true); // Start loading
    const formData = imageUrls.map((url, index) => {
      const parts = url.split("/"); // Split the URL to extract filename and directory
      const filename = parts[parts.length - 1]; // Get the last part as filename
      const directory = parts[parts.length - 2]; // Get the second last part as directory

      return {
        filename,
        directory,
        rating: ratings[index],
      };
    });

    try {
      const response = await fetch("/api/submit", {
        method: "POST",
        headers: {
          Accept: "application/json",
          "Content-Type": "application/json",
        },
        body: JSON.stringify(formData),
      });

      const content = await response.json();

      if (response.ok) {
        toast({
          className: "bg-primary text-white",
          variant: "default",
          title: "Data Submitted",
          description: "Your data has been successfully sent!",
          duration: 5000,
        });

        // Refresh the page after submission
        setTimeout(() => {
          window.location.reload();
        }, 2000); // Delay page reload to show toast
      } else {
        toast({
          className: "bg-red-600 text-white",
          title: "Submission Error",
          description: `Error: ${content.error}`,
          duration: 5000,
          action: <ToastAction altText="Retry">Retry</ToastAction>,
        });
      }
    } catch (error) {
      console.error("Error submitting form:", error);
      toast({
        className: "bg-red-600 text-white",
        title: "Network Error",
        description: "An error occurred while submitting your data.",
        duration: 5000,
        action: <ToastAction altText="Retry">Retry</ToastAction>,
      });
    } finally {
      setLoading(false); // Stop loading
    }
  };

  // Use useEffect to simulate button press every 5 seconds
  // useEffect(() => {
  //   const intervalId = setInterval(() => {
  //     handleFinish(); // Trigger handleFinish every 5 seconds
  //   }, 5000);

  //   return () => clearInterval(intervalId); // Clean up on component unmount
  // }, [imageUrls, ratings]); // Depend on imageUrls and ratings to make sure form data is ready

  return (
    <div className="flex min-h-screen flex-col pb-12 pt-12">
      <div className="mb-8 w-full">
        <h2 className="text-center text-xl font-semibold">
          Your score will help us with our research!
        </h2>
        {/* <Image src="/images/cartoongan/5.jpg" alt="banner" width={512} height={512} /> */}
      </div>
      <div className="flex min-h-screen flex-wrap justify-center gap-8">
        {imageUrls.map((src, index) => (
          <div
            key={index}
            className="transform transition-transform duration-200 hover:scale-105"
          >
            <Form
              src={src}
              index={index}
              rating={ratings[index]}
              onRatingChange={handleRatingChange}
            />
          </div>
        ))}
      </div>
      <Button
        onClick={handleFinish}
        className="mx-auto mt-8 w-full max-w-xs"
        disabled={loading}
      >
        {loading ? <div className="spinner"></div> : "Finish"}
      </Button>
    </div>
  );
}

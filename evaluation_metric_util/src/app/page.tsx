"use client";

import { useState, useEffect } from "react";
import { Form } from "@/components/primitive/form";
import { Button } from "@/components/ui/button";
import { useToast } from "@/hooks/use-toast";
import { ToastAction } from "@/components/ui/toast";

export default function Home() {
  const [imageUrls, setImageUrls] = useState<string[]>([]);
  const [ratings, setRatings] = useState<string[]>([]);
  const [currentIndex, setCurrentIndex] = useState(0); // Track current card index
  const [loading, setLoading] = useState(false);
  const [isFinished, setIsFinished] = useState(false); // Track if submission is completed
  const { toast } = useToast();

  useEffect(() => {
    const fetchImages = async () => {
      try {
        const response = await fetch("/api/images");

        if (!response.ok) {
          throw new Error("Failed to fetch images");
        }

        const images = await response.json();
        setImageUrls(images);
        setRatings(Array(images.length).fill("1"));
      } catch (error) {
        console.error("Error fetching images:", error);
      }
    };

    fetchImages();
  }, []);

  const handleRatingChange = (index: number, rating: string) => {
    const updatedRatings = [...ratings];
    updatedRatings[index] = rating;
    setRatings(updatedRatings);
  };

  const handleNext = () => {
    if (currentIndex < imageUrls.length - 1) {
      setCurrentIndex((prev) => prev + 1);
    }
  };

  const handlePrevious = () => {
    if (currentIndex > 0) {
      setCurrentIndex((prev) => prev - 1);
    }
  };

  const handleFinish = async () => {
    setLoading(true);
    const formData = imageUrls.map((url, index) => {
      const parts = url.split("/");
      const filename = parts[parts.length - 1];
      const directory = parts[parts.length - 2];

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

      if (response.ok) {
        toast({
          className: "bg-primary text-white",
          title: "Data Submitted! This window will reload in a moment.",
          description: "Your data has been successfully sent! You're all done!",
        });
        setIsFinished(true); // Mark as finished after successful submission

        // Reload the page after a delay to allow toast display
        setTimeout(() => window.location.reload(), 4000);
      } else {
        toast({
          className: "bg-red-600 text-white",
          title: "Submission Error",
          description: "An error occurred during submission.",
          action: <ToastAction altText="Retry">Retry</ToastAction>,
        });
      }
    } catch (error) {
      console.error("Error submitting form:", error);
      toast({
        className: "bg-red-600 text-white",
        title: "Network Error",
        description: "An error occurred while submitting your data.",
        action: <ToastAction altText="Retry">Retry</ToastAction>,
      });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex min-h-screen flex-col items-center justify-center">
      {imageUrls.length > 0 && (
        <div className="flex min-h-screen w-full flex-col items-center justify-center">
          <Form
            src={imageUrls[currentIndex]}
            index={currentIndex}
            rating={ratings[currentIndex]}
            onRatingChange={(index, rating) =>
              handleRatingChange(index, rating)
            }
          />

          <div className="mt-4 flex space-x-4">
            <Button onClick={handlePrevious} disabled={currentIndex === 0}>
              Previous
            </Button>
            <Button
              onClick={handleNext}
              disabled={currentIndex === imageUrls.length - 1}
            >
              Next
            </Button>
          </div>
          {currentIndex === imageUrls.length - 1 && (
            <Button
              onClick={handleFinish}
              className="mt-4"
              disabled={loading || isFinished} // Disable when loading or finished
            >
              {loading ? "Submitting..." : isFinished ? "Finished" : "Finish"}
            </Button>
          )}
        </div>
      )}
    </div>
  );
}

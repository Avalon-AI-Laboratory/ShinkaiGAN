"use client";

import Image from "next/image";
import { Label } from "@/components/ui/label";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import {
  Card,
  CardHeader,
  CardTitle,
  CardContent,
  CardFooter,
} from "@/components/ui/card";

interface FormProps {
  src: string;
  index: number;
  rating: string;
  onRatingChange: (index: number, rating: string) => void;
}

export const Form = ({ src, index, rating, onRatingChange }: FormProps) => {
  return (
    <Card className="mx-auto max-w-md shadow-lg">
      {/* Card Header */}
      <CardHeader>
        <CardTitle className="text-center text-2xl font-semibold text-gray-800">
          Rate Image {index + 1}
        </CardTitle>
        <p className="mt-2 text-center text-sm text-gray-500">
          Please rate the image from 1 to 5
        </p>
      </CardHeader>

      {/* Card Content */}
      <CardContent className="flex flex-col items-center">
        {/* Image */}
        <div className="mb-6">
          <div className="overflow-hidden rounded-md shadow-md">
            <Image
              src={src}
              alt={`Sample Image ${index + 1}`}
              width={512}
              height={512}
              className="object-cover"
            />
          </div>
        </div>

        {/* Rating radio buttons */}
        <RadioGroup
          value={rating}
          onValueChange={(value) => onRatingChange(index, value)}
        >
          <div className="mb-4 flex justify-center gap-6">
            {[...Array(5)].map((_, idx) => {
              const value = (idx + 1).toString();
              return (
                <div key={value} className="flex items-center space-x-2">
                  <RadioGroupItem
                    value={value}
                    id={`option-${index}-${value}`}
                    className="transition-all focus:ring-2 focus:ring-blue-500"
                  />
                  <Label
                    htmlFor={`option-${index}-${value}`}
                    className="text-gray-700"
                  >
                    {value}
                  </Label>
                </div>
              );
            })}
          </div>
        </RadioGroup>
      </CardContent>

      {/* Card Footer */}
      <CardFooter className="flex flex-col items-center">
        <p className="text-gray-700">
          Selected Rating: <span className="font-bold">{rating}</span>
        </p>
        <p className="text-xs text-gray-400">(1 = Lowest, 5 = Highest)</p>
      </CardFooter>
    </Card>
  );
};

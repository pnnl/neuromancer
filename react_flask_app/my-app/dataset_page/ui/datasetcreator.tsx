"use client";

import React, { useState } from 'react';
import { Button } from "@/components/ui/button"
import { DropdownMenuTrigger, DropdownMenuLabel, DropdownMenuSeparator, DropdownMenuRadioItem, DropdownMenuRadioGroup, DropdownMenuContent, DropdownMenu } from "@/components/ui/dropdown-menu"
import { Input } from "@/components/ui/input"
import DraggableSquare from '@/components/ui/draggablesquare'; // Import DraggableSquare
import EnvironmentPane from '@/components/environment_pane';



export function DatasetCreatorPage({ toggleDatasetView }) {
  const [environmentSquares, setEnvironmentSquares] = useState([]);
  const [selectedSystem, setSelectedSystem] = useState("lotka-volterra");
  const [numberOfSimulations, setNumberOfSimulations] = useState("");

  const handleGenerateDataset = () => {
    const newSquare = {
      id: `square_${environmentSquares.length + 1}`,
      name: selectedSystem,
      position: { x: 0, y: 0 }
    };
    setEnvironmentSquares(prevState => [...prevState, newSquare]);
  };

  return (
    <div className="grid grid-cols-2 gap-4">
      {/* Left side: Dataset Creator */}
      <div className="col-span-1">
        <div className="grid gap-4">
          <div className="flex items-center gap-4">
            <DropdownMenu>
              {/* System dropdown */}
              <DropdownMenuTrigger asChild>
                <Button className="w-[180px] justify-between text-left" size="sm" variant="outline">
                  <span className="truncate">{selectedSystem}</span>
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent>
                <DropdownMenuLabel>System</DropdownMenuLabel>
                <DropdownMenuSeparator />
                <DropdownMenuRadioGroup defaultValue={selectedSystem} onChange={setSelectedSystem}>
                  <DropdownMenuRadioItem value="lotka-volterra">Lotka-Volterra</DropdownMenuRadioItem>
                  <DropdownMenuRadioItem value="lorenz">Lorenz</DropdownMenuRadioItem>
                  <DropdownMenuRadioItem value="pendulum">Pendulum</DropdownMenuRadioItem>
                </DropdownMenuRadioGroup>
              </DropdownMenuContent>
            </DropdownMenu>
            <Input
              className="w-[200px]"
              placeholder="Enter number of simulations"
              value={numberOfSimulations}
              onChange={e => setNumberOfSimulations(e.target.value)}
            />
            {/* Generate Dataset button */}
            <Button className="h-9" onClick={handleGenerateDataset}>Generate Dataset</Button>
          </div>
        </div>
      </div>
      {/* Right side: EnvironmentPane */}
      <div className="col-span-1">
        <EnvironmentPane environmentSquares={environmentSquares} />
      </div>
      <div className="flex justify-center col-span-2">
        {/* Button to toggle back to Main view */}
        <Button onClick={toggleDatasetView}>Components View</Button>
      </div>
    </div>
  );
}
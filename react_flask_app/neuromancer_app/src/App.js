/**
 * v0 by Vercel.
 * @see https://v0.dev/t/KXbYVG69jUt
 * Documentation: https://v0.dev/docs#integrating-generated-code-into-your-nextjs-app
 */
import Link from "next/link"
import { Button } from "@/components/ui/button"
import { DropdownMenuTrigger, DropdownMenuLabel, DropdownMenuSeparator, DropdownMenuItem, DropdownMenuContent, DropdownMenu } from "@/components/ui/dropdown-menu"

export default function Component() {
  return (
    <div className="flex h-screen w-full flex-col">
      <div className="hidden border-r bg-gray-900 lg:block dark:bg-gray-950">
        <div className="flex h-full flex-col gap-2">
          <div
            className="flex h-[60px] items-center border-b px-6"
            style={{
              backgroundColor: "navy",
            }}
          >
            <Link className="flex items-center gap-2 font-semibold text-black" href="#">
              <Package2Icon className="h-6 w-6" />
              <span className="">Neuromancer</span>
            </Link>
            <Button
              className="ml-auto h-8 w-8"
              size="icon"
              style={{
                backgroundColor: "navy",
              }}
              variant="outline"
            >
              <BellIcon className="h-4 w-4 text-black" />
              <span className="sr-only">Toggle notifications</span>
            </Button>
          </div>
          <div className="flex-1 overflow-auto py-2">
            <nav className="grid items-start px-4 text-sm font-medium">
              <Link className="text-white" href="#">
                Foo
              </Link>
              <Link className="text-white" href="#">
                Bar
              </Link>
            </nav>
          </div>
          <div className="mt-auto p-4">
            <div className="flex items-center justify-center w-full">
              <div className="flex items-center justify-center w-1/2">
                <Button
                  className="w-full"
                  style={{
                    backgroundColor: "navy",
                  }}
                >
                  Dataset View
                </Button>
              </div>
            </div>
          </div>
        </div>
      </div>
      <div className="flex flex-1 flex-col">
        <header
          className="flex h-14 lg:h-[60px] items-center gap-4 border-b bg-gray-900 px-6 dark:bg-gray-950"
          style={{
            backgroundColor: "navy",
          }}
        >
          <Link className="lg:hidden" href="#">
            <Package2Icon className="h-6 w-6 text-black" />
            <span className="sr-only">Home</span>
          </Link>
          <nav className="hidden lg:flex h-8 w-full shrink-0 items-center gap-4 text-sm font-medium lg:gap-8 lg:text-base" />
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button
                className="rounded-full border border-gray-200 w-8 h-8 dark:border-gray-800"
                size="icon"
                variant="ghost"
              >
                <img
                  alt="Avatar"
                  className="rounded-full"
                  height="32"
                  src="/placeholder.svg"
                  style={{
                    aspectRatio: "32/32",
                    objectFit: "cover",
                  }}
                  width="32"
                />
                <span className="sr-only">Toggle user menu</span>
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end">
              <DropdownMenuLabel>My Account</DropdownMenuLabel>
              <DropdownMenuSeparator />
              <DropdownMenuItem>Settings</DropdownMenuItem>
              <DropdownMenuItem>Support</DropdownMenuItem>
              <DropdownMenuSeparator />
              <DropdownMenuItem>Logout</DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        </header>
        <main className="flex flex-1 flex-col gap-4 p-4 md:gap-8 md:p-6">
          <div className="flex h-full items-center justify-center border-dashed border-2 border-gray-200/50 border-gray-200/50 rounded-lg dark:border-gray-800/50">
            <div className="flex items-center justify-center w-full h-full bg-gray-100 dark:bg-gray-800">
              <div className="flex items-center justify-center w-3/4 h-3/4 bg-white dark:bg-gray-900 rounded-lg shadow-lg">
                <div className="flex items-center justify-center w-1/4 h-full bg-gray-200 dark:bg-gray-700 rounded-l-lg">
                  <h2 className="text-lg font-semibold text-black">Components</h2>
                  <div className="flex flex-wrap gap-2 p-2">
                    <div className="w-8 h-8 bg-gray-300 dark:bg-gray-700 rounded" />
                    <div className="w-8 h-8 bg-gray-300 dark:bg-gray-700 rounded" />
                    <div className="w-8 h-8 bg-gray-300 dark:bg-gray-700 rounded" />
                  </div>
                </div>
                <div className="flex items-center justify-center w-3/4 h-full bg-gray-50 dark:bg-gray-800 rounded-r-lg">
                  <h2 className="text-lg font-semibold text-black">Canvas</h2>
                </div>
              </div>
            </div>
          </div>
        </main>
      </div>
    </div>
  )
}

function BellIcon(props) {
  return (
    <svg
      {...props}
      xmlns="http://www.w3.org/2000/svg"
      width="24"
      height="24"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="M6 8a6 6 0 0 1 12 0c0 7 3 9 3 9H3s3-2 3-9" />
      <path d="M10.3 21a1.94 1.94 0 0 0 3.4 0" />
    </svg>
  )
}


function Package2Icon(props) {
  return (
    <svg
      {...props}
      xmlns="http://www.w3.org/2000/svg"
      width="24"
      height="24"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="M3 9h18v10a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V9Z" />
      <path d="m3 9 2.45-4.9A2 2 0 0 1 7.24 3h9.52a2 2 0 0 1 1.8 1.1L21 9" />
      <path d="M12 3v6" />
    </svg>
  )
}

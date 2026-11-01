module Main (main) where

main :: IO ()
main = putStrLn ("hello " ++ show (sum [i * i | i <- [1 .. 10 :: Int]]))

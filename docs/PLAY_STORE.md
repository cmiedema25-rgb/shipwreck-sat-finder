# Play upload path

Android project builds an AAB.
Not published from this repo.

## Build
cd android then run gradlew bundleRelease
Output under app/build/outputs/bundle/release/
Copy AAB to dist/

## Console
App id com.cmiedema25.shipwrecksatfinder
Internal testing first
Host privacy policy from docs/PRIVACY_POLICY.md
No claim of live store listing



## Signing
Create upload keystore offline; never commit secrets.
Configure signingConfigs in android/app/build.gradle or keystore.properties (gitignored).

## Privacy
Host docs/PRIVACY_POLICY.md at a public URL for Play Console.

## Honesty
Do not claim the app is already on Google Play from this repository alone.

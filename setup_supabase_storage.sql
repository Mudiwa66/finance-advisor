-- Supabase Storage Policies Setup
-- FIRST: Go to Supabase Dashboard → Storage and create a bucket named "bank-statements" (set to Private)
-- THEN: Run this SQL in the SQL Editor

-- Storage policies for bank-statements bucket
CREATE POLICY "Users can upload own statements"
ON storage.objects FOR INSERT
WITH CHECK (
    bucket_id = 'bank-statements'
    AND (storage.foldername(name))[1] = current_setting('app.current_user_hash', TRUE)
);

CREATE POLICY "Users can view own statements"
ON storage.objects FOR SELECT
USING (
    bucket_id = 'bank-statements'
    AND (storage.foldername(name))[1] = current_setting('app.current_user_hash', TRUE)
);

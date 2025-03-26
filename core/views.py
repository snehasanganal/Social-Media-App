from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from django.shortcuts import render, redirect
from django.contrib.auth.models import User,auth
from django.contrib import messages
from django.http import HttpResponse, JsonResponse 
from django.contrib.auth.decorators import login_required
from .models import Profile,Post,LikePost,FollowersCount
from itertools import chain
import random
import os
import torch
from django.shortcuts import render, redirect
from transformers import BertTokenizer
from .binary.init import PrivacyBERTLSTM, extract_entities
from .multiclass.init import PrivacyBERTLSTM_MultiClass

#  Get the base directory of the Django project
#  Define the model directory path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, 'core', 'ai_models')

BINARY_MODEL_PATH = os.path.join(MODEL_DIR, "best_binary_lr_1e-05_batch_32_opt_AdamW.pt")
MULTICLASS_MODEL_PATH = os.path.join(MODEL_DIR, "best_multi_lr_1e-05_batch_16_opt_RMSprop.pt")

#  Load Tokenizer
TOKENIZER = "bert-base-cased"
tokenizer = BertTokenizer.from_pretrained(TOKENIZER)

#  Load Binary Model
binary_model = PrivacyBERTLSTM(learning_rate=1e-5, batch_size=32, optimizer_choice="Adam")
binary_model.load_state_dict(torch.load(BINARY_MODEL_PATH, map_location=torch.device("cpu")))
binary_model.eval()

#  Load Multiclass Model
multiclass_model = PrivacyBERTLSTM_MultiClass(learning_rate=1e-5, batch_size=16, optimizer_choice="RMSprop")
multiclass_model.load_state_dict(torch.load(MULTICLASS_MODEL_PATH, map_location=torch.device("cpu")))
multiclass_model.eval()

#  Define Labels
binary_labels = ["Other", "Sensitive"]
multiclass_labels = ["Health", "Politics", "Religion", "Sexuality","Location","Personal Information"]

#  Function to Preprocess Text
"""def preprocess_text(text):
    tokens = tokenizer(text, padding="max_length", max_length=50, truncation=True, return_tensors="pt")
    return tokens["input_ids"], tokens["attention_mask"]"""

def preprocess_text(text):
    tokens = tokenizer(text, padding="max_length", max_length=50, truncation=True, return_tensors="pt")
    entity_flag = extract_entities(text)  # Get entity-aware feature
    return tokens["input_ids"], tokens["attention_mask"], torch.tensor(entity_flag).float().unsqueeze(0)

#  Function to Predict Sensitivity Using Hierarchical Model
def predict_sensitivity(text):
    #input_ids, attention_mask = preprocess_text(text)
    input_ids, attention_mask, entity_flags = preprocess_text(text)
    #  Step 1: Binary Classification
    with torch.no_grad():
        binary_output = binary_model(input_ids, attention_mask,entity_flags)
        binary_pred = torch.argmax(binary_output, dim=1).item()

    if binary_pred == 0:  # "Other" (Non-Sensitive)
        return "Other"

    #  Step 2: If Sensitive, Use Multiclass Model
    with torch.no_grad():
        multiclass_output = multiclass_model(input_ids, attention_mask)
        multiclass_pred = torch.argmax(multiclass_output, dim=1).item()

    return multiclass_labels[multiclass_pred]

import re

def mask_sensitive_info(text):
    """Mask personal information in the input text."""

    # Mask phone numbers (10 digits)
    text = re.sub(r'\b\d{10}\b', '[PHONE]', text)

    # Mask email addresses
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', text)

    # Mask credit/debit card numbers (16-digit format with/without spaces or hyphens)
    text = re.sub(r'\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b', '[CARD]', text)

    # Mask bank account numbers (basic format 9-18 digits)
    text = re.sub(r'\b\d{9,18}\b', '[BANK_ACC]', text)

    # Mask passport numbers (alphanumeric with 8-9 chars)
    text = re.sub(r'\b[A-Z0-9]{8,9}\b', '[PASSPORT]', text)

    # Mask PAN card (10 alphanumeric characters, starting with 5 letters)
    text = re.sub(r'\b[A-Z]{5}[0-9]{4}[A-Z]\b', '[PAN]', text)

    # Mask Aadhaar number (12-digit pattern)
    text = re.sub(r'\b\d{4}[- ]?\d{4}[- ]?\d{4}\b', '[AADHAAR]', text)

    # Mask IFSC codes (4 letters + 0 + 6 alphanumeric characters)
    text = re.sub(r'\b[A-Z]{4}0[A-Z0-9]{6}\b', '[IFSC]', text)

    # Mask SSN (Social Security Number, US format)
    text = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '[SSN]', text)

    # Mask DOB in different formats (DD/MM/YYYY, YYYY-MM-DD, 3rd March, 2007, etc.)
    text = re.sub(r'\b\d{1,2}(st|nd|rd|th)?\s*(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s*\d{4}\b', '[DOB]', text)
    text = re.sub(r'\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b', '[DOB]', text)
    text = re.sub(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', '[DOB]', text)

    # Handle phrases like "born in 2009" or "born on March 3, 2007"
    text = re.sub(r'\bborn (on|in) (\d{4}|\d{1,2}(st|nd|rd|th)?\s*[A-Za-z]+\s*\d{4})\b', '[DOB]', text, flags=re.IGNORECASE)

    # Mask age with different phrases and formats
    text = re.sub(r'\b(?:[Ii] am|[Mm]y age is|[Aa]ge:?)\s?\d{1,3}\s?(years? old)?\b', '[AGE]', text)
    text = re.sub(r'\b(?:aged|age of|[Aa]ge)\s?\d{1,3}\b', '[AGE]', text)
    text = re.sub(r'\b\d{1,3}\s?(years? old)\b', '[AGE]', text)

    # Handle new age variations like "turning 56 today", "will be 45 next year"
    text = re.sub(r'\bturning\s\d{1,3}\s(today|this year|soon)\b', '[AGE]', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(?:will be|going to be|turns|turn)\s\d{1,3}\s(next year|tomorrow|soon)\b', '[AGE]', text, flags=re.IGNORECASE)

    # Mask person names and locations (basic approach using word lists)
    names_list = ["John", "Jane", "Michael", "Alice", "Robert", "Emily"]  # Add common names or use NLP
    locations_list = ["New York", "London", "Paris", "Mumbai", "Sydney"]

    for name in names_list:
        text = re.sub(fr'\b{name}\b', '[NAME]', text, flags=re.IGNORECASE)

    for location in locations_list:
        text = re.sub(fr'\b{location}\b', '[LOCATION]', text, flags=re.IGNORECASE)

    return text

# Create your views here.
@login_required(login_url='signin')
def upload_page(request):
    return render(request, 'upload.html')

@login_required(login_url='signin')
def upload_post(request):
    
    if request.method == 'POST':
        user = request.user.username
        caption = request.POST.get('caption', '').strip()
        override = request.POST.get("override", "false") == "true"
        mask_info = request.POST.get("mask_info", "false") == "true"
        
        # ✅ Handling Image Upload
        image = request.FILES.get("image_upload")
        saved_image = request.POST.get("saved_image", "")
        
        # ✅ Image Handling
        if not image and saved_image:
            image_path = saved_image  
        elif image:
            image_path = default_storage.save(f"post_images/{image.name}", ContentFile(image.read()))
        else:
            image_path = None
        
        # ✅ Sensitivity Check
        sensitivity_result = predict_sensitivity(caption) if caption else "Other"
        
        # ✅ Nudge Logic
        if sensitivity_result != "Other" and not override:
            request.session["saved_image"] = image_path  # Store image path in session
            return render(request, "upload.html", {
                "nudge": f"{sensitivity_result}",
                "caption": caption,
                "show_nudge": True,
            })
        
        # ✅ Masking Logic for Personal Information
        if sensitivity_result == "Personal Information" and mask_info:
            caption = mask_sensitive_info(caption)
        
        # ✅ Save Post
        new_post = Post.objects.create(user=user, caption=caption, image=image_path)
        new_post.save()
        request.session.pop("saved_image", None)  # Clear session data
        return redirect("/")
    
    return redirect("/")
"""def upload_post(request):
    
    if request.method == 'POST':
        user = request.user.username
        caption = request.POST.get('caption', '').strip()
        override = request.POST.get("override", "false") == "true"
        
        # ✅ Handling Image Upload
        image = request.FILES.get("image_upload")
        saved_image = request.POST.get("saved_image", "")

        # ✅ Image Handling
        if not image and saved_image:
            image_path = saved_image  
        elif image:
            image_path = default_storage.save(f"post_images/{image.name}", ContentFile(image.read()))
        else:
            image_path = None

        # ✅ Sensitivity Check
        sensitivity_result = predict_sensitivity(caption) if caption else "Other"

        # ✅ Nudge Logic
        if sensitivity_result != "Other" and not override:
            request.session["saved_image"] = image_path  # Store image path in session
            return render(request, "upload.html", {
                "nudge": f"{sensitivity_result}",
                "caption": caption,
                "show_nudge": True,
                
            })
        
        
        # ✅ Save Post
        new_post = Post.objects.create(user=user, caption=caption, image=image_path)
        new_post.save()

        request.session.pop("saved_image", None)  # Clear session data
        return redirect("/")

    return redirect("/")"""


@login_required(login_url='signin')
def index(request):
    user_object = User.objects.get(username=request.user.username)
    user_profile = Profile.objects.get(user=user_object)
    
    user_following_list = []
    feed = []


    user_following = FollowersCount.objects.filter(follower=request.user.username)

    for users in user_following:
        user_following_list.append(users.user)
    for usernames in user_following_list:
        feed_lists = Post.objects.filter(user=usernames)
        feed.append(feed_lists)
    
    feed_list = list(chain(*feed))

   #user suggestions starts
    all_users = User.objects.all()
    user_following_all=[]

    for user in user_following:
        user_list = User.objects.get(username=user.user)
        user_following_all.append(user_list)

    new_suggestions_list = [x for x in list(all_users) if (x not in list(user_following_all))]
    current_user = User.objects.filter(username=request.user.username)
    final_suggestions_list =[x for x in list(new_suggestions_list) if (x not in list(current_user))]
    random.shuffle(final_suggestions_list)

    username_profile = []
    username_profile_list = []

    for users in final_suggestions_list:
        username_profile.append(users.id)
    
    for ids in username_profile:
        profile_lists = Profile.objects.filter(id_user=ids)
        username_profile_list.append(profile_lists)
    
    suggestions_username_profile_list = list(chain(*username_profile_list))


    return render(request, 'index.html', {'user_profile':user_profile, 'posts': feed_list, 'suggestions_username_profile_list':suggestions_username_profile_list[:4]})


@login_required(login_url='signin')
def search(request):
    user_object = User.objects.get(username=request.user.username)
    user_profile = Profile.objects.get(user=user_object)

    if request.method == 'POST':
        username = request.POST['username']
        username_object = User.objects.filter(username__icontains=username)
        username_profile = []
        username_profile_list = []

        for users in username_object:
            username_profile.append(users.id)
        
        for ids in username_profile:
            profile_lists = Profile.objects.filter(id_user=ids)
            username_profile_list.append(profile_lists)

        username_profile_list = list(chain(*username_profile_list))
    return render(request,'search.html', {'user_profile':user_profile,'username_profile_list':username_profile_list})



@login_required(login_url='signin')
def like_post(request):
    username = request.user.username
    post_id = request.GET.get('post_id')

    post = Post.objects.get(id=post_id)

    like_filter = LikePost.objects.filter(post_id=post_id, username=username).first()

    if like_filter == None:
        new_like = LikePost.objects.create(post_id=post_id, username=username)
        new_like.save()
        post.no_of_likes = post.no_of_likes + 1
        post.save()
        return redirect('/')
    else:
        like_filter.delete()
        post.no_of_likes = post.no_of_likes-1
        post.save()
        return redirect('/')

@login_required(login_url='signin')
def profile(request, pk):
    user_object = User.objects.get(username=pk)
    user_profile = Profile.objects.get(user=user_object)
    user_posts = Post.objects.filter(user=pk)
    user_post_length = len(user_posts)

    follower = request.user.username
    user = pk

    if FollowersCount.objects.filter(follower=follower, user=user).first():
        button_text = 'Unfollow'
    else:
        button_text = 'Follow'

    user_followers = len(FollowersCount.objects.filter(user=pk))
    user_following = len(FollowersCount.objects.filter(follower=pk))
    context={
        'user_object': user_object,
        'user_profile': user_profile,
        'user_posts': user_posts,
        'user_post_length': user_post_length,
        'button_text':button_text,
        'user_followers':user_followers,
        'user_following':user_following
    }
    return render(request, 'profile.html', context )

@login_required(login_url='signin')
def follow(request):
    if request.method == 'POST':
        follower = request.POST['follower']
        user = request.POST['user']

        if FollowersCount.objects.filter(follower=follower, user=user).first():
            delete_follower = FollowersCount.objects.get(follower=follower, user=user)
            delete_follower.delete()
            return redirect('/profile/'+user)
        else:
            new_follower = FollowersCount.objects.create(follower=follower, user=user)
            new_follower.save()
            return redirect('/profile/'+user)


    else:
        return redirect('/')

@login_required(login_url='signin')
def settings(request):
    user_profile = Profile.objects.get(user=request.user)
    if request.method == 'POST':

        if request.FILES.get('image') == None:
            image = user_profile.profileimg
            bio = request.POST['bio']
            location = request.POST['location']

            user_profile.profileimg = image
            user_profile.bio = bio
            user_profile.location = location
            user_profile.save()

        if request.FILES.get('image') != None:
            image = request.FILES.get('image')
            bio = request.POST['bio']
            location = request.POST['location']

            user_profile.profileimg = image
            user_profile.bio = bio
            user_profile.location = location
            user_profile.save()

        return redirect('settings')

    
    return render(request, 'setting.html',{'user_profile': user_profile})

def signup(request):

    if request.method == 'POST':
        username = request.POST['username']
        email = request.POST['email']
        password = request.POST['password']
        password2 = request.POST['password2']

        if password == password2:
            if User.objects.filter(email=email).exists():
                messages.info(request,'Email Taken')
                return redirect('signup')
            elif User.objects.filter(username=username).exists():
                messages.info(request,'Username Taken')
                return redirect('signup')
            else:
                user = User.objects.create_user(username=username, email=email, password=password)
                user.save()

                #log user in and redirect to settings page
                user_login = auth.authenticate(username=username, password=password)
                auth.login(request, user_login)


                #create a Profile object for the new user
                user_model = User.objects.get(username=username)
                new_profile = Profile.objects.create(user=user_model, id_user=user_model.id)
                new_profile.save()
                return redirect('settings')
        else:
            messages.info(request,'Password Not Matching')
            return redirect('signup')

    else:
        return render(request, 'signup.html')
    
def signin(request):
    
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']

        user = auth.authenticate(username=username, password=password)

        if user is not None:
            auth.login(request, user)
            return redirect('/')
        else:
            messages.info(request, 'Credentials Invalid')
            return redirect('signin')
    else:
        return render(request, 'signin.html')
    
@login_required(login_url='signin')
def logout(request):
    auth.logout(request)
    return redirect('signin')
# Идемпотентно создаёт пользователей, организацию и членства.
# Запускается ВНУТРИ контейнера cvat_server через:
#   docker exec -i cvat_server python3 /home/django/manage.py shell < create_users.py
# Параметры берутся из переменных окружения (передаём их в docker exec).

import os

from django.contrib.auth import get_user_model
from django.utils import timezone

from cvat.apps.organizations.models import Membership, Organization

User = get_user_model()


def ensure_user(username, password, email, is_superuser=False):
    user, created = User.objects.get_or_create(
        username=username,
        defaults={"email": email},
    )
    user.email = email
    user.is_active = True
    user.is_staff = is_superuser
    user.is_superuser = is_superuser
    user.set_password(password)  # каждый запуск синхронизирует пароль с .env
    user.save()

    # Отмечаем e-mail подтверждённым (на случай ACCOUNT_EMAIL_VERIFICATION!="none").
    try:
        from allauth.account.models import EmailAddress

        EmailAddress.objects.update_or_create(
            user=user,
            email=email,
            defaults={"verified": True, "primary": True},
        )
    except Exception as exc:  # noqa: BLE001
        print(f"  (пропущено создание EmailAddress: {exc})")

    print(f"  пользователь {username!r}: {'создан' if created else 'обновлён'} "
          f"(superuser={is_superuser})")
    return user


def ensure_membership(user, organization, role):
    membership, created = Membership.objects.get_or_create(
        user=user,
        organization=organization,
        defaults={"role": role, "is_active": True, "joined_date": timezone.now()},
    )
    changed = False
    if membership.role != role:
        membership.role = role
        changed = True
    if not membership.is_active:
        membership.is_active = True
        changed = True
    if membership.joined_date is None:
        membership.joined_date = timezone.now()
        changed = True
    if changed:
        membership.save()
    print(f"  членство {user.username!r} в {organization.slug!r}: роль={role} "
          f"({'создано' if created else 'обновлено'})")


admin_user = os.environ["ADMIN_USER"]
admin_pass = os.environ["ADMIN_PASS"]
ann_user = os.environ["ANN_USER"]
ann_pass = os.environ["ANN_PASS"]
org_slug = os.environ["ORG_SLUG"]
org_name = os.environ["ORG_NAME"]

print("Пользователи:")
admin = ensure_user(admin_user, admin_pass, "admin@example.com", is_superuser=True)
annotator = ensure_user(ann_user, ann_pass, "user@example.com", is_superuser=False)

org, created = Organization.objects.get_or_create(
    slug=org_slug,
    defaults={"name": org_name, "owner": admin},
)
if org.name != org_name or org.owner_id != admin.id:
    org.name = org_name
    org.owner = admin
    org.save()
print(f"Организация {org_slug!r} ({org_name!r}): {'создана' if created else 'обновлена'}")

print("Членства:")
ensure_membership(admin, org, Membership.OWNER)
ensure_membership(annotator, org, Membership.WORKER)

print("create_users.py: готово.")
